"""
grider.py
"""

import numpy as np
from opt_einsum import contract
from pylibxc import LibXCFunctional as Functional

try:
    import psi4
    has_psi4 = True
except ImportError:
    has_psi4 = False

if has_psi4:
    class Psi4Grider():
        """
        Psi4 Grider
        Handles components on the grid using Psi4 grid blocks. 
        """
        def __init__(self, mol, basis, ref):
            
            self.mol = mol
            self.ref = ref
            self.basis = basis
            self.basis_str = basis.name()
            self.nbf   = self.basis.nbf()

            wfn_base = psi4.core.Wavefunction.build(self.mol, self.basis_str)
            self.wfn = psi4.proc.scf_wavefunction_factory('svwn', wfn_base, "UKS")
            self.wfn.initialize()

            self.nalpha = self.wfn.nalpha()
            self.nbeta  = self.wfn.nbeta()

            # Clean Vpot
            restricted = True if ref == 1 else False
            reference  = "RV" if ref == 1 else "UV"
            functional = psi4.driver.dft.build_superfunctional("SVWN", restricted=restricted)[0]
            self.Vpot = psi4.core.VBase.build(self.basis, functional, reference)
            self.Vpot.initialize()

            # self.spherical_points = np.array(self.Vpot.get_np_xyzw())[:3,:].T
            # self.w = np.array(self.Vpot.get_np_xyzw())[3,:].T

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

        def generate_grid(self, x, y, z):
            """
            Genrates Mesh from 3 separate linear spaces and flatten,
            needed for cubic grid.
            Parameters
            ----------
            grid: tuple of three np.ndarray
                (x, y, z)
            Returns
            -------
            grid: np.ndarray
                shape (3, len(x)*len(y)*len(z)).
            """
            # x,y,z, = grid
            shape = (len(x), len(y), len(z))
            X,Y,Z = np.meshgrid(x, y, z, indexing='ij')
            X = X.reshape((X.shape[0] * X.shape[1] * X.shape[2], 1))
            Y = Y.reshape((Y.shape[0] * Y.shape[1] * Y.shape[2], 1))
            Z = Z.reshape((Z.shape[0] * Z.shape[1] * Z.shape[2], 1))
            grid = np.concatenate((X,Y,Z), axis=1).T

            return grid, shape

        def dft_grid_to_fock(self, value, Vpot=None):
                """For value on DFT spherical grid, Fock matrix is returned.
                VFock_ij = \int dx \phi_i(x) \phi_j(x) value(x)
                
                Parameters:
                -----------
                value: np.ndarray of shape (npoint, ).
                Vpot:psi4.core.VBase
                    Vpotential object with info about grid.
                    Provides DFT spherical grid. Only comes to play if no grid is given.
                
                Returns:
                ---------
                VFock: np.ndarray of shape (nbasis, nbasis)
                """
                if Vpot is None:
                    Vpot = self.Vpot
                VFock = np.zeros((self.nbf, self.nbf))
                points_func = Vpot.properties()[0]

                i = 0
                # Loop over the blocks
                for b in range(Vpot.nblocks()):
                    # Obtain block information
                    block = Vpot.get_block(b)
                    points_func.compute_points(block)
                    npoints = block.npoints()
                    lpos = np.array(block.functions_local_to_global())
                    if len(lpos) == 0:
                        i += npoints
                        continue
                    # Obtain the grid weight
                    w = np.array(block.w())

                    # Compute phi!
                    phi = np.array(points_func.basis_values()["PHI"])[:npoints, :lpos.shape[0]]

                    Vtmp = np.einsum('pb,p,p,pa->ab', phi, value[i:i+npoints], w, phi, optimize=True)

                    # Add the temporary back to the larger array by indexing, ensure it is symmetric
                    VFock[(lpos[:, None], lpos)] += 0.5 * (Vtmp + Vtmp.T)

                    i += npoints
                assert i == value.shape[0], "Did not run through all the points. %i %i" %(i, value.shape[0])
                return VFock

        def ao(self, coeff, grid=None, basis=None):
            """
            Generates a quantity on the grid given its ao representation.
            *This is the most general function for basis to grid transformation.

            Parameters
            ----------
            coeff: np.ndarray
                Vector/Matrix of quantity on ao basis. Shape: {(num_ao_basis, ), (num_ao_basis, num_ao_basis)}
            grid: np.ndarray Shape: (3, npoints) or (4, npoints) or tuple for block_handler (return of grid_to_blocks)
                grid where density will be computed.
            basis: psi4.core.BasisSet, optional
                The basis set. If not given it will use target wfn.basisset().
            Vpot: psi4.core.VBase
                Vpotential object with info about grid. 
                Provides DFT spherical grid. Only comes to play if no grid is given. 

            Returns
            -------
            coeff_r: np.ndarray Shape: (npoints, )
                Quantity expressed by the coefficient on the given grid 


            """


            if grid is not None:
                if type(grid) is np.ndarray:
                    if grid.shape[0] != 3 and grid.shape[0] != 4:
                        raise ValueError("The shape of grid should be (3, npoints) "
                                        "or (4, npoints) but got (%i, %i)" % (grid.shape[0], grid.shape[1]))
                    blocks, npoints, points_function = self.grid_to_blocks(grid, basis=basis)
                else:
                    blocks, npoints, points_function = grid
            elif grid is None:
                Vpot = self.Vpot
                nblocks = Vpot.nblocks()
                blocks = [Vpot.get_block(i) for i in range(nblocks)]
                npoints = Vpot.grid().npoints()
                points_function = Vpot.properties()[0]
            else:
                raise ValueError("A grid or a V_potential (DFT grid) must be given.")

            coeff_r = np.zeros((npoints))

            offset = 0
            for i_block in blocks:
                points_function.compute_points(i_block)
                b_points = i_block.npoints()
                offset += b_points
                lpos = np.array(i_block.functions_local_to_global())
                if len(lpos)==0:
                    continue
                phi = np.array(points_function.basis_values()["PHI"])[:b_points, :lpos.shape[0]]

                if coeff.ndim == 1:
                    l_mat = coeff[(lpos[:])]
                    coeff_r[offset - b_points : offset] = contract('pm,m->p', phi, l_mat)
                elif coeff.ndim == 2:
                    l_mat = coeff[(lpos[:, None], lpos)]
                    coeff_r[offset - b_points : offset] = contract('pm,mn,pn->p', phi, l_mat, phi)

            return coeff_r

        def density(self, Da, Db=None, grid=None, Vpot=None):
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
            if Vpot is None:
                Vpot = self.Vpot

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
                nblocks = Vpot.nblocks()
                blocks = [Vpot.get_block(i) for i in range(nblocks)]
                npoints = Vpot.grid().npoints()
                points_function = Vpot.properties()[0]
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

        def esp(self, Da=None, Db=None, vpot=None, grid=None, compute_hartree=True, wfn=None):
            """
            Generates EXTERNAL/ESP/HARTREE and Fermi Amaldi Potential on given grid
            Parameters
            ----------
            Da,Db: np.ndarray, opt, shape (nbf, nbf)
                The electron density in the denominator of Hartee potential. If None, the original density matrix
                will be used.
            grid: np.ndarray Shape: (3, npoints) or (4, npoints) or tuple for block_handler (return of grid_to_blocks)
                grid where density will be computed.
            vpot: psi4.core.VBase
                Vpotential object with info about grid.
                Provides DFT spherical grid. Only comes to play if no grid is given. 
            
            Returns
            -------
            vext, hartree, esp: np.ndarray
                External, Hartree, and Full Electronstatic potential on the given grid
                Shape: (npoints, )
            """
            if wfn is None:
                wfn = self.wfn

            if Da is not None or Db is not None:
                Da_temp = np.copy(self.wfn.Da().np)
                Db_temp = np.copy(self.wfn.Db().np)
                if Da is not None:
                    wfn.Da().np[:] = Da
                if Db is not None:
                    wfn.Db().np[:] = Db

            nthreads = psi4.get_num_threads()
            psi4.set_num_threads(1)
            
            if grid is not None:
                if type(grid) is np.ndarray:
                    blocks, npoints, points_function = self.grid_to_blocks(grid)
                else:
                    blocks, npoints, points_function = grid
            elif vpot is not None:
                nblocks = vpot.nblocks()
                blocks = [vpot.get_block(i) for i in range(nblocks)]
                npoints = vpot.grid().npoints()

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

            if compute_hartree:
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

                    #EXTERNAL
                    for atom in range(natoms):
                        r =  np.sqrt((x-rs[atom][0])**2 + (y-rs[atom][1])**2 + (z-rs[atom][2])**2)
                        vext_temp = - 1.0 * zs[atom] / r
                        vext_temp[np.isinf(vext_temp)] = 0.0
                        vext[offset - b_points : offset] += vext_temp

                    if compute_hartree:
                        #ESP
                        xyz = np.concatenate((x[:,None],y[:,None],z[:,None]), axis=1)
                        grid_block = psi4.core.Matrix.from_array(xyz)
                        esp[offset - b_points : offset] = esp_wfn.compute_esp_over_grid_in_memory(grid_block).np

            #Hartree
            if compute_hartree:
                hartree = - 1.0 * (vext + esp)

            if Da is not None:
                self.wfn.Da().np[:] = Da_temp
            if Db is not None:
                self.wfn.Db().np[:] = Db_temp
            psi4.set_num_threads(nthreads)

            if compute_hartree:
                return vext, hartree, esp
            else:
                return vext

        def orbitals(self, Ca=None, Cb=None, grid=None, Vpot=None):
            """
            Generates orbitals given grid
            Parameters
            ----------
            Ca, Cb: np.ndarray
                Alpha, Beta Orbital Coefficient Matrix. Shape: (num_ao_basis, num_ao_basis)
            grid: np.ndarray Shape: (3, npoints) or (4, npoints) or tuple for block_handler (return of grid_to_blocks)
                grid where density will be computed
            Vpot: psi4.core.VBase
                Vpotential object with info about grid.
                Provides DFT spherical grid. Only comes to play if no grid is given. 
            Returns
            -------
            orbitals: np.ndarray
                Orbitals on the given grid of size . 
                Shape: (nbasis, npoints, ref)
            """
            if Vpot is None:
                Vpot = self.Vpot

            if Ca is None and Cb is None:
                Ca = psi4.core.Matrix.from_array(self.Ca)
                Cb = psi4.core.Matrix.from_array(self.Cb)
            else:
                Ca = psi4.core.Matrix.from_array(Ca)
                Cb = psi4.core.Matrix.from_array(Cb)

            if self.ref == 2 and Cb is None:
                raise ValueError("Db is required for an unrestricted system")

            if grid is not None:
                if type(grid) is np.ndarray:
                    if grid.shape[0] != 3 and grid.shape[0] != 4:
                        raise ValueError("The shape of grid should be (3, npoints) "
                                        "or (4, npoints) but got (%i, %i)" % (grid.shape[0], grid.shape[1]))
                    blocks, npoints, points_function = self.grid_to_blocks(grid)
                else:
                    blocks, npoints, points_function = grid
            elif grid is None and Vpot is not None:
                nblocks = Vpot.nblocks()
                blocks = [Vpot.get_block(i) for i in range(nblocks)]
                npoints = Vpot.grid().npoints()
                points_function = Vpot.properties()[0]
            else:
                raise ValueError("A grid or a V_potential (DFT grid) must be given.")

            if self.ref == 1:
                orbitals_r = [np.zeros((npoints)) for i_orb in range(self.nbf)]
                points_function.set_pointers(Ca)
                Ca_np = Ca.np
            if self.ref == 2:
                orbitals_r = [np.zeros((npoints, 2)) for i_orb in range(self.nbf)]
                points_function.set_pointers(Ca, Cb)
                Ca_np = Ca.np
                Cb_np = Cb.np

            offset = 0
            for i_block in blocks:
                points_function.compute_points(i_block)
                b_points = i_block.npoints()
                offset += b_points
                lpos = np.array(i_block.functions_local_to_global())
                if len(lpos)==0:
                    continue
                phi = np.array(points_function.basis_values()["PHI"])[:b_points, :lpos.shape[0]]

                for i_orb in range(self.nbf):
                    Ca_local = Ca_np[lpos, i_orb]
                    if self.ref == 1:
                        orbitals_r[i_orb][offset - b_points : offset] = contract('m, pm -> p', Ca_local, phi)
                    else:
                        Cb_local = Cb_np[lpos, i_orb]
                        orbitals_r[i_orb][offset - b_points : offset,0] = contract('m, pm -> p', Ca_local, phi)
                        orbitals_r[i_orb][offset - b_points : offset,1] = contract('m, pm -> p', Cb_local, phi)
            return orbitals_r

        def lap_phi(self, Ca=None, Cb=None, grid=None, Vpot=None):
            """
            Generates laplacian of molecular orbitals
            Parameters
            ----------
        Ca, Cb: np.ndarray
                Alpha, Beta Orbital Coefficient Matrix. Shape: (num_ao_basis, num_ao_basis)
            grid: np.ndarray Shape: (3, npoints) or (4, npoints) or tuple for block_handler (return of grid_to_blocks)
                grid where density will be computed.
            Vpot: psi4.core.VBase
                Vpotential object with info about grid.
                Provides DFT spherical grid. Only comes to play if no grid is given. 
            Returns
            -------
            lap_phi: List[np.ndarray]. Where array is of shape (npoints, ref)
                Laplacian of molecular orbitals on the grid
            """
            if Vpot is None:
                Vpot = self.Vpot

            if Ca is None and Cb is None:
                Ca = psi4.core.Matrix.from_array(self.Ca)
                Cb = psi4.core.Matrix.from_array(self.Cb)
            else:
                Ca = psi4.core.Matrix.from_array(Ca)
                Cb = psi4.core.Matrix.from_array(Cb)

            if self.ref == 2 and Cb is None:
                raise ValueError("Db is required for an unrestricted system")

            if grid is not None:
                if type(grid) is np.ndarray:
                    if grid.shape[0] != 3 and grid.shape[0] != 4:
                        raise ValueError("The shape of grid should be (3, npoints) "
                                        "or (4, npoints) but got (%i, %i)" % (grid.shape[0], grid.shape[1]))
                    blocks, npoints, points_function = self.grid_to_blocks(grid)
                else:
                    blocks, npoints, points_function = grid
            elif grid is None and Vpot is not None:
                nblocks = Vpot.nblocks()
                blocks = [Vpot.get_block(i) for i in range(nblocks)]
                npoints = Vpot.grid().npoints()
                points_function = Vpot.properties()[0]
            else:
                raise ValueError("A grid or a V_potential (DFT grid) must be given.")

            points_function.set_ansatz(2)

            if self.ref == 1:
                points_function.set_pointers(Ca)
                lap_phi = [np.zeros((npoints)) for i_orb in range(self.nbf)]
            else:
                points_function.set_pointers(Ca, Cb)
                lap_phi = [np.zeros((npoints, 2)) for i_orb in range(self.nbf)]

            offset = 0
            for i_block in blocks:
                points_function.compute_points(i_block)
                b_points = i_block.npoints()
                offset += b_points
                lpos = np.array(i_block.functions_local_to_global())
                if len(lpos)==0:
                    continue
                
                #Obtain subset of phi_@@ matrices
                lx = np.array(points_function.basis_values()["PHI_XX"])[:b_points, :lpos.shape[0]]
                ly = np.array(points_function.basis_values()["PHI_YY"])[:b_points, :lpos.shape[0]]
                lz = np.array(points_function.basis_values()["PHI_ZZ"])[:b_points, :lpos.shape[0]]

                for i_orb in range(self.nbf):
                    Ca_local = Ca.np[lpos, i_orb][:,None]
                    
                    if self.ref ==1:
                        lap_phi[i_orb][offset - b_points : offset] += ((lx + ly + lz) @ Ca_local)[:,0]
                    else:
                        Cb_local = Cb.np[lpos, i_orb][:,None]
                        lap_phi[i_orb][offset - b_points : offset, 0] += ((lx + ly + lz) @ Ca_local)[:,0]
                        lap_phi[i_orb][offset - b_points : offset, 1] += ((lx + ly + lz) @ Cb_local)[:,0]

            return lap_phi

        def grad_phi(self, Ca=None, Cb=None, grid=None,Vpot=None):
            """
            Generates laplacian of molecular orbitals
            Parameters
            ----------
        Ca, Cb: np.ndarray
                Alpha, Beta Orbital Coefficient Matrix. Shape: (num_ao_basis, num_ao_basis)
            grid: np.ndarray Shape: (3, npoints) or (4, npoints) or tuple for block_handler (return of grid_to_blocks)
                grid where density will be computed.
            Vpot: psi4.core.VBase
                Vpotential object with info about grid.
                Provides DFT spherical grid. Only comes to play if no grid is given. 
            Returns
            -------
            grad_phi: List[np.ndarray]. Where array is of shape (npoints, ref)
                Gradient of molecular orbitals on the grid
            """
            if Vpot is None:
                Vpot = self.Vpot

            if Ca is None and Cb is None:
                Ca = psi4.core.Matrix.from_array(self.Ca)
                Cb = psi4.core.Matrix.from_array(self.Cb)
            else:
                Ca = psi4.core.Matrix.from_array(Ca)
                Cb = psi4.core.Matrix.from_array(Cb)

            if self.ref == 2 and Cb is None:
                raise ValueError("Db is required for an unrestricted system")

            if grid is not None:
                if type(grid) is np.ndarray:
                    if grid.shape[0] != 3 and grid.shape[0] != 4:
                        raise ValueError("The shape of grid should be (3, npoints) "
                                        "or (4, npoints) but got (%i, %i)" % (grid.shape[0], grid.shape[1]))
                    blocks, npoints, points_function = self.grid_to_blocks(grid)
                else:
                    blocks, npoints, points_function = grid
            elif grid is None and Vpot is not None:
                nblocks = Vpot.nblocks()
                blocks = [Vpot.get_block(i) for i in range(nblocks)]
                npoints = Vpot.grid().npoints()
                points_function = Vpot.properties()[0]
            else:
                raise ValueError("A grid or a V_potential (DFT grid) must be given.")

            points_function.set_ansatz(2)

            if self.ref == 1:
                points_function.set_pointers(Ca)
                grad_phi = [np.zeros((npoints)) for i_orb in range(self.nbf)]
            else:
                points_function.set_pointers(Ca, Cb)
                grad_phi = [np.zeros((npoints, 2)) for i_orb in range(self.nbf)]

            offset = 0
            for i_block in blocks:
                points_function.compute_points(i_block)
                b_points = i_block.npoints()
                offset += b_points
                lpos = np.array(i_block.functions_local_to_global())
                if len(lpos)==0:
                    continue
                
                #Obtain subset of phi_@ matrix
                gx = np.array(points_function.basis_values()["PHI_X"])[:b_points, :lpos.shape[0]]
                gy = np.array(points_function.basis_values()["PHI_Y"])[:b_points, :lpos.shape[0]]
                gz = np.array(points_function.basis_values()["PHI_Z"])[:b_points, :lpos.shape[0]]

                for i_orb in range(self.nbf):
                    Ca_local = Ca.np[lpos, i_orb][:,None]
                    if self.ref == 1:
                        grad_phi[i_orb][offset - b_points : offset] += ((gx + gy + gz) @ Ca_local)[:,0]
                    if self.ref == 2:
                        Cb_local = Cb.np[lpos, i_orb][:,None]
                        grad_phi[i_orb][offset - b_points : offset, 0] += ((gx + gy + gz) @ Ca_local)[:,0]
                        grad_phi[i_orb][offset - b_points : offset, 1] += ((gx + gy + gz) @ Cb_local)[:,0]

            return grad_phi

        def vxc(self, func_id=1, grid=None, Da=None, Db=None, Vpot=None):
            """
            Generates Vxc given grid
            Parameters
            ----------
            Da, Db: np.ndarray
                Alpha, Beta densities. Shape: (num_ao_basis, num_ao_basis)
            func_id: int
                Functional ID associated with Density Functional Approximationl.
                Full list of functionals: https://www.tddft.org/programs/libxc/functionals/
            grid: np.ndarray Shape: (3, npoints) or (4, npoints) or tuple for block_handler (return of grid_to_blocks)
                grid where density will be computed.
            Vpot: psi4.core.VBase
                Vpotential object with info about grid.
                Provides DFT spherical grid. Only comes to play if no grid is given. 
            Returns
            -------
            VXC: np.ndarray
                Exchange correlation potential on the given grid
                Shape: (npoints, )
            """

            local_functionals = [1,546,549,532,692,641,552,287,307,578,5,24,4,579,308,289,551,
                                22,23,14,11,574,573,554,5900,12,13,25,9,10,27,3,684,683,17,7,
                                28,29,30,31,8,317,2,6,536,537,538,318,577,259,547,548,20,599,43,
                                51,580,50,550
                                ]

            if func_id not in local_functionals:
                raise ValueError("Only local functionals are supported on the grid")

            if Vpot is None:
                Vpot = self.Vpot

            if grid is not None:
                if type(grid) is np.ndarray:
                    blocks, npoints, points_function = self.grid_to_blocks(grid)
                else:
                    blocks, npoints, points_function = grid
                if self.ref == 1:
                    density = self.density(Da=Da, grid=grid)
                else:
                    density = self.density(Da=Da, Db=Db, grid=grid)
            elif grid is None and Vpot is not None:
                nblocks = Vpot.nblocks()
                blocks = [Vpot.get_block(i) for i in range(nblocks)]
                npoints = Vpot.grid().npoints()
                if self.ref == 1:
                    density = self.density(Da=Da),
                else:
                    density = self.density(Da=Da, Db=Db)
            else:
                raise ValueError("A grid or a V_potential (DFT grid) must be given.")

            ingredients = {'rho' : density}
            if self.ref == 1:
                functional = Functional(func_id, 1)
            else:
                functional = Functional(func_id, 2) 
            xc_dictionary = functional.compute(ingredients)
            vxc = xc_dictionary['vrho']

            return np.squeeze(vxc)

        # --------------------------------------------------------------------------

        def _average_local_orbital_energy(self, D, C, eig, grid_info=None, Vpot=None):
            """
            (4)(6) in mRKS.
            """
            if Vpot is None:
                Vpot = self.Vpot

            if grid_info is None:
                e_bar = np.zeros(Vpot.grid().npoints())
                nblocks = Vpot.nblocks()

                points_func = Vpot.properties()[0]
                points_func.set_deriv(0)
                blocks = None
            else:
                blocks, npoints, points_func = grid_info
                e_bar = np.zeros(npoints)
                nblocks = len(blocks)

                points_func.set_deriv(0)

            # For unrestricted
            iw = 0
            for l_block in range(nblocks):
                # Obtain general grid information
                if blocks is None:
                    l_grid = Vpot.get_block(l_block)
                else:
                    l_grid = blocks[l_block]
                l_npoints = l_grid.npoints()

                points_func.compute_points(l_grid)
                l_lpos = np.array(l_grid.functions_local_to_global())
                if len(l_lpos) == 0:
                    iw += l_npoints
                    continue
                l_phi = np.array(points_func.basis_values()["PHI"])[:l_npoints, :l_lpos.shape[0]]
                lD = D[(l_lpos[:, None], l_lpos)]
                lC = C[l_lpos, :]
                rho = contract('pm,mn,pn->p', l_phi, lD, l_phi)
                e_bar[iw:iw + l_npoints] = contract("pm,mi,ni,i,pn->p", l_phi, lC, lC, eig, l_phi) / rho

                iw += l_npoints
            assert iw == e_bar.shape[0], "Somehow the whole space is not fully integrated."
            return e_bar

        def _get_l_kinetic_energy_density_directly(self, D, C, grid_info=None, Vpot=None):
            """
            Calculate $\frac{\tau_L^{KS}}{\rho^{KS}}-\frac{\tau_P^{KS}}{\rho^{KS}}$:
            laplace_rho_temp: $\frac{\nabla^2 \rho}{4}$;
            tauW_temp: $\frac{|\napla \rho|^2}{8|\rho|}$;
            tauLmP_rho: $\frac{|\napla \rho|^2}{8|\rho|^2} - \frac{\nabla^2 \rho}{4\rho}$.

            (i.e. the 2dn and 3rd term in eqn. (17) in [1] over $\rho$.):
            """

            if Vpot is None:
                Vpot = self.Vpot

            if grid_info is None:
                tauLmP_rho = np.zeros(Vpot.grid().npoints())
                nblocks = Vpot.nblocks()
                points_func = Vpot.properties()[0]
                blocks = None

            else:
                blocks, npoints, points_func = grid_info
                tauLmP_rho = np.zeros(npoints)
                nblocks = len(blocks)

            points_func.set_deriv(2)

            iw = 0
            for l_block in range(nblocks):
                # Obtain general grid information
                if blocks is None:
                    l_grid = Vpot.get_block(l_block)
                else:
                    l_grid = blocks[l_block]
                l_npoints = l_grid.npoints()

                points_func.compute_points(l_grid)
                l_lpos = np.array(l_grid.functions_local_to_global())
                if len(l_lpos) == 0:
                    iw += l_npoints
                    continue
                l_phi = np.array(points_func.basis_values()["PHI"])[:l_npoints, :l_lpos.shape[0]]
                l_phi_x = np.array(points_func.basis_values()["PHI_X"])[:l_npoints, :l_lpos.shape[0]]
                l_phi_y = np.array(points_func.basis_values()["PHI_Y"])[:l_npoints, :l_lpos.shape[0]]
                l_phi_z = np.array(points_func.basis_values()["PHI_Z"])[:l_npoints, :l_lpos.shape[0]]
                l_phi_xx = np.array(points_func.basis_values()["PHI_XX"])[:l_npoints, :l_lpos.shape[0]]
                l_phi_yy = np.array(points_func.basis_values()["PHI_YY"])[:l_npoints, :l_lpos.shape[0]]
                l_phi_zz = np.array(points_func.basis_values()["PHI_ZZ"])[:l_npoints, :l_lpos.shape[0]]

                lD = D[(l_lpos[:, None], l_lpos)]
                # lC = C[l_lpos, :]

                rho = contract('pm,mn,pn->p', l_phi, lD, l_phi)
                rho_inv = 1/rho

                # Calculate the second term
                laplace_rho_temp = contract('ab,pa,pb->p', lD, l_phi, l_phi_xx + l_phi_yy + l_phi_zz)
                # laplace_rho_temp += contract('pm, mn, pn->p', l_phi_x,lD, l_phi_x)
                # laplace_rho_temp += contract('pm, mn, pn->p', l_phi_y,lD, l_phi_y)
                # laplace_rho_temp += contract('pm, mn, pn->p', l_phi_z,lD, l_phi_z)
                laplace_rho_temp += np.sum((l_phi_x @ lD) * l_phi_x, axis=1)
                laplace_rho_temp += np.sum((l_phi_y @ lD) * l_phi_y, axis=1)
                laplace_rho_temp += np.sum((l_phi_z @ lD) * l_phi_z, axis=1)

                laplace_rho_temp *= 0.25 * 2

                # Calculate the third term
                tauW_temp = contract('pm, mn, pn->p', l_phi, lD, l_phi_x) ** 2
                tauW_temp += contract('pm, mn, pn->p', l_phi, lD, l_phi_y) ** 2
                tauW_temp += contract('pm, mn, pn->p', l_phi, lD, l_phi_z) ** 2
                tauW_temp *= rho_inv * 0.125 * 4

                tauLmP_rho[iw: iw + l_npoints] = (-laplace_rho_temp + tauW_temp) * rho_inv
                iw += l_npoints
            assert iw == tauLmP_rho.shape[0], "Somehow the whole space is not fully integrated."

            return tauLmP_rho

        def _pauli_kinetic_energy_density(self, D, C, occ=None, grid_info=None, Vpot=None):
            """
            (16)(18) in mRKS. But notice this does not return taup but taup/n
            :return:
            """
            if Vpot is None:
                Vpot = self.Vpot

            if occ is None:
                occ = np.ones(C.shape[1])

            if grid_info is None:
                taup_rho = np.zeros(Vpot.grid().npoints())
                nblocks = Vpot.nblocks()

                points_func = Vpot.properties()[0]
                points_func.set_deriv(1)
                blocks = None

            else:
                blocks, npoints, points_func = grid_info
                taup_rho = np.zeros(npoints)
                nblocks = len(blocks)
                points_func.set_deriv(1)

            iw = 0
            for l_block in range(nblocks):
                # Obtain general grid information
                if blocks is None:
                    l_grid = Vpot.get_block(l_block)
                else:
                    l_grid = blocks[l_block]
                l_npoints = l_grid.npoints()

                points_func.compute_points(l_grid)
                l_lpos = np.array(l_grid.functions_local_to_global())
                if len(l_lpos) == 0:
                    iw += l_npoints
                    continue
                l_phi = np.array(points_func.basis_values()["PHI"])[:l_npoints, :l_lpos.shape[0]]
                l_phi_x = np.array(points_func.basis_values()["PHI_X"])[:l_npoints, :l_lpos.shape[0]]
                l_phi_y = np.array(points_func.basis_values()["PHI_Y"])[:l_npoints, :l_lpos.shape[0]]
                l_phi_z = np.array(points_func.basis_values()["PHI_Z"])[:l_npoints, :l_lpos.shape[0]]

                lD = D[(l_lpos[:, None], l_lpos)]

                rho = contract('pm,mn,pn->p', l_phi, lD, l_phi)

                lC = C[l_lpos, :]
                # Matrix Methods
                part_x = contract('pm,mi,nj,pn->ijp', l_phi, lC, lC, l_phi_x)
                part_y = contract('pm,mi,nj,pn->ijp', l_phi, lC, lC, l_phi_y)
                part_z = contract('pm,mi,nj,pn->ijp', l_phi, lC, lC, l_phi_z)
                part1_x = (part_x - np.transpose(part_x, (1, 0, 2))) ** 2
                part1_y = (part_y - np.transpose(part_y, (1, 0, 2))) ** 2
                part1_z = (part_z - np.transpose(part_z, (1, 0, 2))) ** 2


                occ_matrix = np.expand_dims(occ, axis=1) @ np.expand_dims(occ, axis=0)

                taup = np.sum((part1_x + part1_y + part1_z).T * occ_matrix, axis=(1,2)) * 0.5

                taup_rho[iw:iw + l_npoints] = taup / rho ** 2 * 0.5

                iw += l_npoints
            assert iw == taup_rho.shape[0], "Somehow the whole space is not fully integrated."
            return taup_rho

