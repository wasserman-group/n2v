"""
grider.py
"""

import numpy as np
import psi4

class Psi4Grider():


    def __init__(self, mol, basis, ref):
        
        self.mol = mol
        self.ref = ref
        self.basis = basis
        self.basis_str = basis.name()
        self.nbf   = self.basis.nbf()

        wfn_base = psi4.core.Wavefunction.build(self.mol, self.basis_str)
        self.wfn = psi4.proc.scf_wavefunction_factory('svwn', wfn_base, "UKS")
        self.wfn.initialize()

        # Clean Vpot
        restricted = True if ref == 1 else False
        reference  = "RV" if ref == 1 else "UV"
        functional = psi4.driver.dft.build_superfunctional("SVWN", restricted=restricted)[0]
        self.Vpot = psi4.core.VBase.build(self.basis, functional, reference)
        self.Vpot.initialize()

        self.npoints = self.Vpot.grid().npoints()   

    def grid_to_blocks(self, grid, basis=None):
        """
        Generate list of blocks to allocate given grid

        Parameters
        ----------
        grid: np.ndarray
            Grid to be distributed into blocks
            Size: (3, npoints) for homogeneous grid
                  (4, npoints) for inhomogenous grid to account for weights
        basis: psi4.core.BasisSet; optional
            The basis set. If not given, it will use target wfn.basisset().

        Returns
        -------
        blocks: list    
            List with psi4.core.BlockOPoints
        npoints: int
            Total number of points (for one dimension)
        points: psi4.core.{RKS, UKS}
            Points function to set matrices.
        """
        assert (grid.shape[0] == 3) or (grid.shape[0] == 4), """Grid does not have the correct dimensions. \n
                                                              Array must be of size (3, npoints) or (4, npoints)"""
        if_w = grid.shape[0] == 4

        if basis is None:
            basis = self.basis

        epsilon    = psi4.core.get_global_option("CUBIC_BASIS_TOLERANCE")
        extens     = psi4.core.BasisExtents(basis, epsilon)
        max_points = psi4.core.get_global_option("DFT_BLOCK_MAX_POINTS")        
        npoints    = grid.shape[1]
        nblocks = int(np.floor(npoints/max_points))
        blocks = []

        max_functions = 0
        #Run through full blocks
        idx = 0
        for nb in range(nblocks):
            x = psi4.core.Vector.from_array(grid[0][idx : idx + max_points])
            y = psi4.core.Vector.from_array(grid[1][idx : idx + max_points])
            z = psi4.core.Vector.from_array(grid[2][idx : idx + max_points])
            if if_w:
                w = psi4.core.Vector.from_array(grid[3][idx : idx + max_points])
            else:
                w = psi4.core.Vector.from_array(np.zeros(max_points))  # When w is not necessary and not given

            blocks.append(psi4.core.BlockOPoints(x, y, z, w, extens))
            idx += max_points
            max_functions = max_functions if max_functions > len(blocks[-1].functions_local_to_global()) \
                                          else len(blocks[-1].functions_local_to_global())

        #Run through remaining points
        if idx < npoints:
            x = psi4.core.Vector.from_array(grid[0][idx:])
            y = psi4.core.Vector.from_array(grid[1][idx:])
            z = psi4.core.Vector.from_array(grid[2][idx:])
            if if_w:
                w = psi4.core.Vector.from_array(grid[3][idx:])
            else:
                w = psi4.core.Vector.from_array(np.zeros_like(grid[2][idx:]))  # When w is not necessary and not given
            blocks.append(psi4.core.BlockOPoints(x, y, z, w, extens))

            max_functions = max_functions if max_functions > len(blocks[-1].functions_local_to_global()) \
                                          else len(blocks[-1].functions_local_to_global())

        zero_matrix = psi4.core.Matrix(basis.nbf(), basis.nbf())
        if self.ref == 1:
            point_func = psi4.core.RKSFunctions(basis, max_points, max_functions)
            point_func.set_pointers(zero_matrix)
        else:
            point_func = psi4.core.UKSFunctions(basis, max_points, max_functions)
            point_func.set_pointers(zero_matrix, zero_matrix)

        return blocks, npoints, point_func

    def density(self, Da, Db=None, grid=None):
        """
        Generates Density given grid

        Parameters
        ----------
        Da, Db: np.ndarray
            Alpha, Beta densities. Shape: (num_ao_basis, num_ao_basis)
        grid: np.ndarray Shape: (3, npoints) or (4, npoints) or tuple for block_handler (return of grid_to_blocks)
            grid where density will be computed.
        vpot: psi4.core.VBase
            Vpotential object with info about grid.
            Provides DFT spherical grid. Only comes to play if no grid is given. 

        Returns
        -------
        density: np.ndarray Shape: (ref, npoints)
            Density on the given grid. 
        """

        Da = psi4.core.Matrix.from_array(Da)
        if Db is not None:
            Db = psi4.core.Matrix.from_array(Db)

        if grid is not None:
            if type(grid) is np.ndarray:
                if grid.shape[0] != 3 and grid.shape[0] != 4:
                    raise ValueError("The shape of grid should be (3, npoints) "
                                     "or (4, npoints) but got (%i, %i)" % (grid.shape[0], grid.shape[1]))
                blocks, npoints, points_function = self.grid_to_blocks(grid)
            else:
                blocks, npoints, points_function = grid
        elif grid is None:
            nblocks = self.Vpot.nblocks()
            blocks = [self.Vpot.get_block(i) for i in range(nblocks)]
            npoints = self.Vpot.grid().npoints()
            points_function = self.Vpot.properties()[0]
        else:
            raise ValueError("A grid or a V_potential (DFT grid) must be given.")

        if Db is None:
            points_function.set_pointers(Da)
            rho_a = points_function.point_values()["RHO_A"]
            density   = np.zeros((npoints))
        else:
            points_function.set_pointers(Da, Db)
            rho_a = points_function.point_values()["RHO_A"]
            rho_b = points_function.point_values()["RHO_B"]
            density   = np.zeros((npoints, self.ref))

        offset = 0
        for i_block in blocks:
            points_function.compute_points(i_block)
            b_points = i_block.npoints()
            offset += b_points

            if self.ref == 1:
                density[offset - b_points : offset] = rho_a.np[ :b_points]
            else:
                density[offset - b_points : offset, 0] = rho_a.np[ :b_points]
                density[offset - b_points : offset, 1] = rho_b.np[ :b_points]

        return density

        pass

    def esp(self, Da=None, Db=None, grid=None, Vpot=None, wfn=None):

        """
        Generates EXTERNAL/ESP/HARTREE and Fermi Amaldi Potential on given grid

        Parameters
        ----------
        Da,Db: np.ndarray, opt, shape (nbf, nbf)
            The electron density in the denominator of Hartee potential. If None, the original density matrix
            will be used.
        grid: np.ndarray Shape: (3, npoints) or (4, npoints) or tuple for block_handler (return of grid_to_blocks)
            grid where density will be computed.
        Vpot: psi4.core.VBase
            Vpotential object with info about grid.
            Provides DFT spherical grid. Only comes to play if no grid is given. 
        
        Returns
        -------
        vext, hartree, esp, v_fa: np.ndarray
            External, Hartree, ESP, and Fermi Amaldi potential on the given grid
            Shape: (npoints, )
        """

        if wfn is None:
            wfn = self.wfn

        if Da is not None:
            Da_temp = np.copy(wfn.Da().np)
            wfn.Da().np[:] = Da
        if  Db is not None:
            Db_temp = np.copy(wfn.Db().np)
            wfn.Db().np[:] = Db

        nthreads = psi4.get_num_threads()
        psi4.set_num_threads(1)
        
        if grid is not None:
            if type(grid) is np.ndarray:
                blocks, npoints, points_function = self.grid_to_blocks(grid)
            else:
                blocks, npoints, points_function = grid
        elif grid is None:
            Vpot = self.Vpot
            nblocks = Vpot.nblocks()
            blocks = [Vpot.get_block(i) for i in range(nblocks)]
            npoints = Vpot.grid().npoints()
        else:
            raise ValueError("A grid or a V_potential (DFT grid) must be given.")

        #Initialize Arrays
        vext = np.zeros(npoints)
        esp  = np.zeros(npoints)

        #Get Atomic Information
        mol_dict = self.mol.to_schema(dtype='psi4')
        natoms = len(mol_dict["elem"])
        indx = [i for i in range(natoms) if self.mol.charge(i) != 0.0]
        natoms = len(indx)
        #Atomic numbers and Atomic positions
        zs = [mol_dict["elez"][i] for i in indx]
        rs = [self.mol.geometry().np[i] for i in indx]

        esp_wfn = psi4.core.ESPPropCalc(wfn)

        #Loop Through blocks
        offset = 0
        with np.errstate(divide='ignore'):
            for i_block in blocks:
                b_points = i_block.npoints()
                offset += b_points
                x = i_block.x().np
                y = i_block.y().np
                z = i_block.z().np

                # EXTERNAL
                for atom in range(natoms):
                    r =  np.sqrt((x-rs[atom][0])**2 + (y-rs[atom][1])**2 + (z-rs[atom][2])**2)
                    vext_temp = - 1.0 * zs[atom] / r
                    vext_temp[np.isinf(vext_temp)] = 0.0
                    vext[offset - b_points : offset] += vext_temp
                # ESP
                xyz = np.concatenate((x[:,None],y[:,None],z[:,None]), axis=1)
                grid_block = psi4.core.Matrix.from_array(xyz)
                esp[offset - b_points : offset] = esp_wfn.compute_esp_over_grid_in_memory(grid_block).np

        # HARTREE
        hartree = - 1.0 * (vext + esp)

        if Da is not None:
            wfn.Da().np[:] = Da_temp
        if Db is not None:
            wfn.Db().np[:] = Db_temp
        psi4.set_num_threads(nthreads)

        return vext, hartree, esp

    def get_orbitals():
        pass
