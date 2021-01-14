"""
grider.py

Generates grid for plotting 
"""

import numpy as np
from dataclasses import dataclass
import psi4
psi4.core.be_quiet()
from pylibxc import LibXCFunctional as Functional

from time import time

from .cubeprop import Cubeprop

try:
    from rich import print
except:
    pass

import warnings
import matplotlib.pyplot as plt
from opt_einsum import contract

@dataclass
class data_bucket:
    pass

default_grid = np.concatenate((np.linspace(-10,10,20, dtype=np.half)[:,None],
                              np.linspace(-10,10,20, dtype=np.half)[:,None],
                              np.linspace(-10,10,20, dtype=np.half)[:,None]), axis=1)

class Grider(Cubeprop):

    def grid_to_blocks(self, grid):
        """
        Generate list of blocks to allocate given grid

        Parameters
        ----------
        grid: np.ndarray
            Grid to be distributed into blocks
            Size: (npoints, 3) for homogeneous grid
                  (npoints, 4) for inhomogenous grid to account for weights

        Returns
        -------
        blocks: list    
            List with psi4.core.BlockOPoints
        npoints: int
            Total number of points (for one dimension)
        points: psi4.core.{RKS, UKS}
            Points function to set matrices.
        """
        assert (grid.shape[0] == 3) or (grid.shape[0] == 4), "Grid does not have the correct dimensions. \
                                                              Array must be of size (npoints, 3) or (npoints, 4)"
        if_w = grid.shape[0] == 4
             
        epsilon    = psi4.core.get_global_option("CUBIC_BASIS_TOLERANCE")
        extens     = psi4.core.BasisExtents(self.basis, epsilon)
        max_points = psi4.core.get_global_option("DFT_BLOCK_MAX_POINTS")        
        npoints    = grid.shape[1]
        nblocks = int(np.floor(npoints/max_points))
        blocks = []

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

        max_functions = 0 if 0 > len(blocks[-1].functions_local_to_global()) \
                                      else len(blocks[-1].functions_local_to_global())

        if self.ref == 1:
            point_func = psi4.core.RKSFunctions(self.basis, max_points, max_functions)
        else:
            point_func = psi4.core.UKSFunctions(self.basis, max_points, max_functions)

        return blocks, npoints, point_func

    def generate_mesh(self, grid=default_grid):
        """
        Genrates Mesh from 3 separate linear spaces
        needed for cubic grid.

        Parameters
        ----------
        grid: np.ndarray
            Grid needed to be turned into 3D mesh.
        
        Returns
        -------
        out_grid: np.ndarray
            Mesh grid as 3dimensional array
        """

        shape = (grid.shape[1], grid.shape[1], grid.shape[1])
        X,Y,Z = np.meshgrid(grid[0],grid[1],grid[2])
        X = X.reshape((X.shape[0] * X.shape[1] * X.shape[2], 1))
        Y = Y.reshape((Y.shape[0] * Y.shape[1] * Y.shape[2], 1))
        Z = Z.reshape((Z.shape[0] * Z.shape[1] * Z.shape[2], 1))
        grid = np.concatenate((X,Y,Z), axis=1).T

        return grid, shape

    def generate_dft_grid(self, wfn):
        """
        Extracts DFT spherical grid and weights from wfn object

        Parameters
        ----------
        wfn: psi4.core.{RKS, UKS}

        Returns
        -------
        dft_grid: list
            Numpy arrays corresponding to x,y,z, and w.
            Shape: (4, npoints**3)
        
        """

        try:
            vpot = wfn.V_potential()
        except:
            raise ValueError("Wfn object does not contain a Vpotential object. Try with one obtained from a DFT calculation")

        nblocks = vpot.nblocks()
        blocks = [vpot.get_block(i) for i in range(nblocks)]
        npoints = vpot.grid().npoints()
        points_function = vpot.properties()[0]

        x = np.empty((npoints))
        y = np.empty((npoints))
        z = np.empty((npoints))
        w = np.empty((npoints))

        offset = 0
        for i_block in blocks:
            b_points = i_block.npoints()
            offset += b_points

            x[offset - b_points : offset] = i_block.x().np
            y[offset - b_points : offset] = i_block.x().np
            z[offset - b_points : offset] = i_block.x().np
            w[offset - b_points : offset] = i_block.x().np

        dft_grid = [x,y,z,w]

        return dft_grid


    #Quantities on Grid
    def on_grid_ao(self, coeff, grid=None, vpot=None, cubic_grid=False):
        """
        Generates a quantity on the grid given its ao representation

        Parameters
        ----------
        coeff: np.ndarray
            Vector/Matrix of quantity on ao basis. Shape: {(num_ao_basis, ), (num_ao_basis, num_ao_basis)}
        grid: np.ndarray. Shape ()
            Grid where density will be computed.
        cubic_grid: bool    
            If False the resulting array won't be reshaped.
            If True the resulting array will be reshaped as (npoints, nponits, npoints). 
            Where npoints is the number of points for the grid in any one dimension.
        vpot: psi4.core.VBase
            Vpotential object with info about grid. 
            Provides DFT spherical grid. Only comes to play if no grid is given. 

        Returns
        -------
        coeff_r: np.ndarray
            Quantity expressed by the coefficient on the given grid 
            Shape: (npoints) if cubic_grid is False
                   (npoints, npoints, npoints) if cubic_grid is True

        """

        if cubic_grid is True and grid is not None:
            nshape = grid.shape[1]
            grid, shape = self.generate_mesh(grid)
        elif cubic_grid is True and grid is None:
            raise ValueError("'Cubic Grid' requires to explicitly specify grid")

        if grid is not None and vpot is None:
            blocks, npoints, points_function = self.grid_to_blocks(grid)
        elif grid is None and vpot is not None:
            nblocks = vpot.nblocks()
            blocks = [vpot.get_block(i) for i in range(nblocks)]
            npoints = vpot.grid().npoints()
            points_function = vpot.properties()[0]
        elif grid is None and vpot is None:
            blocks, npoints, points_function = self.grid_to_blocks(default_grid)

        psi4_coeff = psi4.core.Matrix.from_array(coeff)
        if self.ref == 1:
            points_function.set_pointers(psi4_coeff)
        elif self.ref == 2:
            points_function.set_pointers(psi4_coeff, psi4_coeff)
        coeff_r = np.empty((npoints)) 

        offset = 0
        for i_block in blocks:
            points_function.compute_points(i_block)
            b_points = i_block.npoints()
            offset += b_points
            lpos = np.array(i_block.functions_local_to_global())
            phi = np.array(points_function.basis_values()["PHI"])[:b_points, :lpos.shape[0]]

            if coeff.ndim == 1:
                l_mat = coeff[(lpos[:, None])]
                coeff_r[offset - b_points : offset] = mat_r = contract('pm,m->p', phi, l_mat)
            elif coeff.ndim == 2: 
                l_mat = coeff[(lpos[:, None], lpos)]
                coeff_r[offset - b_points : offset] = mat_r = contract('pm,mn,pn->p', phi, l_mat, phi)

        if cubic_grid is True:
            coeff_r = np.reshape( coeff_r, (nshape, nshape, nshape))

        return coeff_r

    def on_grid_density(self, grid = None, 
                              Da=None, 
                              Db=None,
                              cubic_grid = False,
                              vpot=None)
        """
        Generates Density given grid

        Parameters
        ----------
        Da, Db: np.ndarray
            Alpha, Beta densities. Shape: (num_ao_basis, num_ao_basis)
        grid: np.ndarray
            Grid where density will be computed
        cubic_grid: bool    
            If False the resulting array won't be reshaped.
            If True the resulting array will be reshaped as (npoints, nponits, npoints). 
            Where npoints is the number of points for the grid in any one dimension.
        vpot: psi4.core.VBase
            Vpotential object with info about grid.
            Provides DFT spherical grid. Only comes to play if no grid is given. 

        Returns
        -------
        density: np.ndarray
            Density on the given grid. 
            Shape: (npoints) if cubic_grid is False
                   (npoints, npoints, npoints) if cubic_grid is True


        """

        if Da is None and Db is None:
            Da = psi4.core.Matrix.from_array(self.Da)
            Db = psi4.core.Matrix.from_array(self.Db)
        else:
            Da = psi4.core.Matrix.from_array(Da)
            Db = psi4.core.Matrix.from_array(Db)

        if self.ref == 2 and Db is None:
            raise ValueError("Db is required for an unrestricted system")

        if cubic_grid is True and grid is not None:
            nshape = grid.shape[1]
            grid, shape = self.generate_mesh(grid)
        elif cubic_grid is True and grid is None:
            raise ValueError("'Cubic Grid' requires to explicitly specify grid")

        if grid is not None and vpot is None:
            blocks, npoints, points_function = self.grid_to_blocks(grid)
        elif grid is None and vpot is not None:
            nblocks = vpot.nblocks()
            blocks = [vpot.get_block(i) for i in range(nblocks)]
            npoints = vpot.grid().npoints()
            points_function = vpot.properties()[0]
        elif grid is None and vpot is None:
            blocks, npoints, points_function = self.grid_to_blocks(default_grid)

        density   = np.empty((npoints, self.ref))
        if self.ref == 1:
            points_function.set_pointers(Da)
            rho_a = points_function.point_values()["RHO_A"]
        if self.ref == 2:
            points_function.set_pointers(Da, Db)
            rho_a = points_function.point_values()["RHO_A"]
            rho_b = points_function.point_values()["RHO_B"]

        offset = 0
        for i_block in blocks:
            points_function.compute_points(i_block)
            b_points = i_block.npoints()
            offset += b_points
            density[offset - b_points : offset, 0] = 0.5 * rho_a.np[ :b_points]
            if self.ref == 2:
                density[offset - b_points : offset, 1] = 0.5 * rho_b.np[ :b_points]

        if cubic_grid == True:
            density = np.reshape(density, (nshape, nshape, nshape, self.ref))

        return density

    def on_grid_orbitals(self, grid=None, Ca=None, Cb=None, vpot=None, cubic_grid=False):
        """
        Generates orbitals given grid

        Parameters
        ----------
        Ca, Cb: np.ndarray
            Alpha, Beta Orbital Coefficient Matrix. Shape: (num_ao_basis, num_ao_basis)
        grid: np.ndarray
            Grid where density will be computed. Shape: {(npoints, 3), (npoints, 4)}
        cubic_grid: bool    
            If False the resulting array won't be reshaped.
            If True the resulting array will be reshaped as (npoints, nponits, npoints). 
            Where npoints is the number of points for the grid in any one dimension.
        vpot: psi4.core.VBase
            Vpotential object with info about grid.
            Provides DFT spherical grid. Only comes to play if no grid is given. 

        Returns
        -------
        orbitals: np.ndarray
            Orbitals on the given grid of size . 
            Shape: (nbasis, npoints, ref) if cubic_grid is False
                   (nbasis, npoints, npoints, npoints, ref) if cubic_grid is True


        """

        if Ca is None and Cb is None:
            Ca = psi4.core.Matrix.from_array(self.Ca)
            Cb = psi4.core.Matrix.from_array(self.Cb)
        else:
            Ca = psi4.core.Matrix.from_array(Ca)
            Cb = psi4.core.Matrix.from_array(Cb)

        if self.ref == 2 and Cb is None:
            raise ValueError("Db is required for an unrestricted system")

        if cubic_grid is True and grid is not None:
            nshape = grid.shape[1]
            grid, shape = self.generate_mesh(grid)
        elif cubic_grid is True and grid is None:
            raise ValueError("'Cubic Grid' requires to explicitly specify grid")

        if grid is not None and vpot is None:
            blocks, npoints, points_function = self.grid_to_blocks(grid)
        elif grid is None and vpot is not None:
            nblocks = vpot.nblocks()
            blocks = [vpot.get_block(i) for i in range(nblocks)]
            npoints = vpot.grid().npoints()
            points_function = vpot.properties()[0]
        elif grid is None and vpot is None:
            blocks, npoints, points_function = self.grid_to_blocks(default_grid)

        orbitals_r = [np.empty((npoints, self.ref)) for i_orb in range(self.nbf)]

        if self.ref == 1:
            points_function.set_pointers(Ca)
        if self.ref == 2:
            points_function.set_pointers(Ca, Cb)

        Ca_np = Ca.np
        Cb_np = Cb.np

        offset = 0
        for i_block in blocks:
            points_function.compute_points(i_block)
            b_points = i_block.npoints()
            offset += b_points
            w = np.array(i_block.w())
            lpos = np.array(i_block.functions_local_to_global())
            phi = np.array(points_function.basis_values()["PHI"])[:b_points, :lpos.shape[0]]

            for i_orb in range(self.nbf):
                Ca_local = Ca_np[lpos, i_orb]
                Cb_local = Cb_np[lpos, i_orb]
                orbitals_r[i_orb][offset - b_points : offset,0] = contract('m, pm -> p', Ca_local, phi)
                orbitals_r[i_orb][offset - b_points : offset,1] = contract('m, pm -> p', Cb_local, phi)

        if cubic_grid is True:
            for i_orb in range(self.nbf):
                orbitals_r[i_orb] = np.reshape( orbitals_r[i_orb], (nshape, nshape, nshape, self.ref))

        return orbitals_r

    def on_grid_esp(self, wfn,grid=None, cubic_grid=False, vpot=None):

        """
        Generates EXTERNAL/ESP/HARTREE and Fermi Amaldi Potential on given grid

        Parameters
        ----------
        wfn: psi4.core.{UKS, RKS, CCWavefunction,...}
            Wavefunction Object 
        grid: np.ndarray
            Grid where density will be computed. Shape: {(npoints, 3), (npoints, 4)}
        cubic_grid: bool    
            If False the resulting array won't be reshaped.
            If True the resulting array will be reshaped as (npoints, nponits, npoints). 
            Where npoints is the number of points for the grid in any one dimension.
        vpot: psi4.core.VBase
            Vpotential object with info about grid.
            Provides DFT spherical grid. Only comes to play if no grid is given. 

        Returns
        -------
        vext, hartree, esp, v_fa: np.ndarray
            External, Hartree, ESP, and Fermi Amaldi potential on the given grid
            Shape: (npoints) if cubic_grid is False
                   (npoints, npoints, npoints) if cubic_grid is True
        """

        if cubic_grid is True and grid is not None:
            nshape = grid.shape[1]
            grid, shape = self.generate_mesh(grid)
        elif cubic_grid is True and grid is None:
            raise ValueError("'Cubic Grid' requires to explicitly specify grid")

        if grid is not None and vpot is None:
            blocks, npoints, points_function = self.grid_to_blocks(grid)
        elif grid is None and vpot is not None:
            nblocks = vpot.nblocks()
            blocks = [vpot.get_block(i) for i in range(nblocks)]
            npoints = vpot.grid().npoints()
            points_function = vpot.properties()[0]
        elif grid is None and vpot is None:
            blocks, npoints, points_function = self.grid_to_blocks(default_grid)

        #Initialize Arrays
        vext = np.empty((npoints, self.ref))
        esp  = np.empty((npoints, self.ref))

        #Get Atomic Information
        mol_dict = self.mol.to_schema(dtype='psi4')
        natoms = len(mol_dict["elem"])
        indx = [i for i in range(natoms) if self.mol.charge(i) != 0.0]
        natoms = len(indx)
        #Atomic numbers and Atomic positions
        zs = [mol_dict["elez"][i] for i in indx]
        rs = [self.mol.geometry().np[i] for i in indx]

        esp_wfn = psi4.core.ESPPropCalc(self.wfn)

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
                    r =  np.sqrt( (x-rs[atom][0])**2 + (y-rs[atom][1])**2+ (z-rs[atom][2])**2)
                    vext[offset - b_points : offset, 0] += -1.0 * zs[atom] / r
                for i in range(len(vext[:,0])):
                    if np.isinf(vext[i,0]) == True:
                        vext[i,0] = 0.0
                #ESP
                xyz = np.concatenate((x[:,None],y[:,None],z[:,None]), axis=1) 
                grid_block = psi4.core.Matrix.from_array(xyz)
                esp[offset - b_points : offset, 0] = esp_wfn.compute_esp_over_grid_in_memory(grid_block).np

        #Hartree
        vext[:,1] = vext[:,0]
        hartree = - 1.0 * (vext + esp)
        v_fa = -1.0 / (self.nalpha + self.nbeta) * hartree

        if cubic_grid is True:
            vext    = np.reshape(vext, (nshape, nshape, nshape, self.ref))
            hartree = np.reshape(hartree, (nshape, nshape, nshape, self.ref))
            esp     = np.reshape(esp, (nshape, nshape, nshape, self.ref))
            v_fa    = np.reshape(v_fa, (nshape, nshape, nshape, self.ref))

        return vext, hartree, v_fa, esp

    def on_grid_vxc(self, func_id=1, grid=None, Da=None, Db=None,
                          cubic_grid=False, vpot=None):
        """
        Generates Vxc given grid

        Parameters
        ----------
        Da, Db: np.ndarray
            Alpha, Beta densities. Shape: (num_ao_basis, num_ao_basis)
        func_id: int
            Functional ID associated with Density Functional Approximationl.
            Full list of functionals: https://www.tddft.org/programs/libxc/functionals/
        grid: np.ndarray
            Grid where density will be computed. Shape: {(npoints, 3), (npoints, 4)}
        cubic_grid: bool    
            If False the resulting array won't be reshaped.
            If True the resulting array will be reshaped as (npoints, nponits, npoints). 
            Where npoints is the number of points for the grid in any one dimension.
        vpot: psi4.core.VBase
            Vpotential object with info about grid.
            Provides DFT spherical grid. Only comes to play if no grid is given. 

        Returns
        -------
        VXC: np.ndarray
            Exchange correlation potential on the given grid
            Shape: (npoints) if cubic_grid is False
                   (npoints, npoints, npoints) if cubic_grid is True

        """

        if func_id != 1:
            raise ValueError("Only LDA fucntionals are supported on the grid")

        if cubic_grid is True and grid is not None:
            nshape = grid.shape[1]
            grid, shape = self.generate_mesh(grid)
        elif cubic_grid is True and grid is None:
            raise ValueError("'Cubic Grid' requires to explicitly specify grid")

        if  Da is None and Db is None and vpot is None:
            density = self.on_grid_density(grid=grid)
        elif Da is not None and Db is None and vpot is None:
            density = self.on_grid_density(grid=grid, Da=Da)
        elif Da is not None and Db is not None and vpot is None:
            density = self.on_grid_density(grid=grid, Da=Da, Db=Db)
        ###
        elif Da is None and Db is None and vpot is not None:
            density = self.on_grid_density(vpot=vpot)        
        elif Da is not None and Db is None and vpot is not None:
            density = self.on_grid_density(Da=Da, vpot=vpot)
        elif Da is not None and Db is not None and vpot is not None:
            density = self.on_grid_density(Da=Da, Db=Db, vpot=vpot)

        if grid is not None and vpot is None:
            blocks, npoints, points_function = self.grid_to_blocks(grid)
        elif grid is None and vpot is not None:
            nblocks = vpot.nblocks()
            blocks = [vpot.get_block(i) for i in range(nblocks)]
            npoints = vpot.grid().npoints()
            points_function = vpot.properties()[0]
        elif grid is None and vpot is None:
            blocks, npoints, points_function = self.grid_to_blocks(default_grid)

        vxc = np.empty((npoints, self.ref))
        ingredients = {}
        offset = 0
        for i_block in blocks:
            b_points = i_block.npoints()
            offset += b_points
            ingredients["rho"] = density[offset - b_points : offset, :]

            if self.ref == 1:
                functional = Functional(1, 1)
            else:
                functional = Functional(1, 2) 
            xc_dictionary = functional.compute(ingredients)
            vxc[offset - b_points : offset, :] = np.squeeze(xc_dictionary['vrho'])

        if cubic_grid is True:
            vxc = np.reshape(vxc, (nshape, nshape, nshape, self.ref))

        return vxc


    ##########################SCRATCH FUNCTIONS###############################

    def all_on_grid(self, grid=default_grid, grid_type="1d"):

        if grid_type == "1d":
            pass
        elif grid_type == "cubic":
            grid, shape = generate_mesh(grid)

        blocks, npoints, points_function = self.grid_to_blocks(grid)
        
        self.grid = data_bucket
        self.c_grid = data_bucket
        density   = np.empty((npoints, self.ref))
        hartree   = np.empty((npoints, self.ref))
        vext      = np.empty((npoints, self.ref))
        esp       = np.empty((npoints, self.ref))
        vxc       = np.empty((npoints, self.ref))

        #GENERATE DENSITY ####################################
        Da = psi4.core.Matrix.from_array(self.Da)
        Db = psi4.core.Matrix.from_array(self.Db)
        if self.ref == 1:
            #?? Set D?
            points_function.set_pointers(Da)
            rho_a = points_function.point_values()["RHO_A"]
        if self.ref == 2:
            #?? Set D?
            points_function.set_pointers(Da, Db)
            rho_a = points_function.point_values()["RHO_A"]
            rho_b = points_function.point_values()["RHO_B"]

        offset = 0
        for i_block in blocks:
            points_function.compute_points(i_block)
            b_points = i_block.npoints()
            offset += b_points
            density[offset - b_points : offset, 0] = 0.5 * rho_a.np[ :b_points]
            if self.ref == 2:
                density[offset - b_points : offset, 1] = 0.5 * rho_b.np[ :b_points]

        self.grid.density = density

        ####################################

        #GENERATE EXTERNAL
        mol_dict = self.mol.to_schema(dtype='psi4')
        natoms = len(mol_dict["elem"])
        indx = [i for i in range(natoms) if self.mol.charge(i) != 0.0]
        natoms = len(indx)
        #Atomic numbers and Atomic positions
        zs = [mol_dict["elez"][i] for i in indx]
        rs = [self.mol.geometry().np[i] for i in indx]

        offset = 0
        with np.errstate(divide='ignore'):
            for i_block in blocks:
                b_points = i_block.npoints()
                offset += b_points
                x = i_block.x().np
                y = i_block.y().np
                z = i_block.z().np
                for atom in range(natoms):
                    r =  np.sqrt( (x-rs[atom][0])**2 + (y-rs[atom][1])**2+ (z-rs[atom][2])**2)
                    vext[offset - b_points : offset, 0] += -1.0 * zs[atom] / r
                for i in range(len(vext[:,0])):
                    if np.isinf(vext[i,0]) == True:
                        vext[i,0] = 0.0
            vext[:,1] = vext[:,0]

        self.grid.vext = vext

         ####################################

        #HARTREE + ESP
        esp_wfn = psi4.core.ESPPropCalc(self.wfn)
        offset = 0
        for i_block in blocks:
            b_points = i_block.npoints()
            offset += b_points
            x = i_block.x().np[:,None]
            y = i_block.y().np[:,None]
            z = i_block.z().np[:,None]
            xyz = np.concatenate((x,y,z), axis=1) 
            grid_block = psi4.core.Matrix.from_array(xyz)
            esp[offset - b_points : offset, 0] = esp_wfn.compute_esp_over_grid_in_memory(grid_block).np

        #Hartree
        hartree = - 1.0 * (vext + esp)
        v_fa = -1.0 / (self.nalpha + self.nbeta) * hartree
        self.grid.hartree = hartree
        self.grid.v_fa = v_fa
        self.grid.esp = esp

        ####VXC################################
        warnings.warn("Only LDA fucntionals are supported on the grid")
        ingredients = {}
        offset = 0
        for i_block in blocks:
            b_points = i_block.npoints()
            offset += b_points
            ingredients["rho"] = density[offset - b_points : offset, :]
            if self.ref == 1:
                functional = Functional(1, 1)
            else:
                functional = Functional(1, 2) 
            xc_dictionary = functional.compute(ingredients)
            vxc[offset - b_points : offset, :] = np.squeeze(xc_dictionary['vrho'])

        self.grid.vxc = vxc
        ####################################

        #Generate 2D information
        if grid_type == "cubic":
            self.c_grid.density = np.reshape(density, (N0, N1, N2, density.shape[1]))
            self.c_grid.vext = np.reshape(density, (N0, N1, N2, vext.shape[1]))
            self.c_grid.hartree = np.reshape(density, (N0, N1, N2, hartree.shape[1]))
            self.c_grid.esp = np.reshape(density, (N0, N1, N2, esp.shape[1]))
            self.c_grid.vxc = np.reshape(density, (N0, N1, N2, vxc.shape[1]))
                
    def build_rectangular_grid(self, 
                               grid = default_grid,
                               return_hartree = False,
                               return_1d      = False,
                               return_2d      = False,
                               reutrn_3d      = False,
                               one_dim = False):


        blocks, npoints, points_function = self.grid_to_blocks(self.grid)


        x = grid[0]
        y = grid[1]
        z = grid[2]

        # GENERATE DENSITY
        density = np.zeros(int(npoints))
        points.set_pointers(psi4.core.Matrix.from_array(self.Da))
        rho = points.point_values()["RHO_A"]
        offset = 0
        for i in range(len(block)):
            points.compute_points(block[i])
            n_points = block[i].npoints()
            offset += n_points
            density[offset-n_points:offset] = 0.5 * rho.np[:n_points]

        lpos = np.array(block[0].functions_local_to_global())
        phi = np.array(points.basis_values()["PHI"])[:npoints, :lpos.shape[0]]
        self.phi = phi

        self.grid = data_bucket
        self.grid.xyz = np.array((grid[0],grid[1],grid[2])).T    
        self.grid.density = density 

        #GENERATE EXTERNAL
        mol_dict = self.mol.to_schema(dtype='psi4')
        natoms = len(mol_dict["elem"])
        indx = [i for i in range(natoms) if self.mol.charge(i) != 0.0]
        natoms = len(indx)
        #Atomic numbers and Atomic positions
        zs = [mol_dict["elez"][i] for i in indx]
        rs = [self.mol.geometry().np[i] for i in indx]

        vext = np.zeros(npoints)
        with np.errstate(divide='ignore'):
            for atom in range(natoms):
                r =  np.sqrt( (x-rs[atom][0])**2 + (y-rs[atom][1])**2+ (z-rs[atom][2])**2)
                vext += -1.0 * zs[atom] / r
        for i in range(len(vext)):
            if np.isinf(vext[i]) == True:
                vext[i] = 0.0

                                    
            self.grid.vext = vext

        #HARTREE
        if return_hartree == True:
            esp_wfn = psi4.core.ESPPropCalc(self.wfn)
            grid_block = psi4.core.Matrix.from_array(self.grid.xyz)
            esp = esp_wfn.compute_esp_over_grid_in_memory(grid_block).np
            self.grid.esp = esp

            #Hartree
            hartree = - 1.0 * (vext + esp)
            self.grid.hartree = hartree

            v_fa = -1.0 / (self.nalpha + self.nbeta) * hartree
            self.grid.v_fa = v_fa

        #VXC
        warnings.warn("Only LDA fucntionals are supported on the grid")
        ingredients = {}
        ingredients["rho"] = density
        functional = Functional(1, 1) 
        xc_dictionary = functional.compute(ingredients)
        vxc = np.squeeze(xc_dictionary['vrho'])

        self.grid.vxc = vxc

        #POTENTIAL FROM OPTIMIZER
        if self.ref == 1:
            vopt_a = contract('pm,m->p', phi, self.v_opt)
            vopt = np.concatenate((vopt_a, vopt_a))
        else:
            vopt_a = contract('pm,m->p', phi, self.v_opt[:self.naux])
            vopt_b = contract('pm,m->p', phi, self.v_opt[self.naux:])
            vopt = np.concatenate(( vopt_a, vopt_b ))
    
        self.grid.vopt = vopt

        #GENERATE ONE DIMENSION
        if one_dim is True:
            indx  = []
            for i in range(len(grid[0])):
                if np.abs(grid[0][i]) < 1e-8:
                    if np.abs(grid[1][i]) < 1e-8:
                        indx.append( i )
            if len(indx) == 0:
                warnings.warn("Warning. Cubic grid is not covering the z axis")
            indx = np.array(indx)

            self.grid.z         = grid[2][indx]
            self.grid.z_density = density[indx]
            self.grid.z_esp     = esp[indx] if return_hartree ==True else None
            self.grid.z_hartree = hartree[indx] if return_hartree ==True else None
            self.grid.z_vfa     = v_fa[indx] if return_hartree ==True else None
            self.grid.z_vext     = vext[indx]
            self.grid.z_vxc      = vxc[indx]
            self.grid.z_vopt     = vopt[indx]

        #GENERANTE 2 DIMENSIONS
        self.grid.cube_density = np.reshape(density, (int(N[0]), int(N[1]), int(N[2])))
        # self.grid.cube_vopt    = np.reshape(vopt, (int(N[0]), int(N[1]), int(N[2])))
        # self.grid.cube_vxc     = np.reshape(vxc, (int(N[0]), int(N[1]), int(N[2])))

        psi4.set_options({"CUBIC_BLOCK_MAX_POINTS" : 1000})
        block, points, phi = None, None, None
        return

    def get_from_grid(self):
    
        mol_grid  = self.mol
        basis_str = self.basis_str
        density_a = self.Da
        density_b = self.Db

        _, wfn = psi4.energy( "svwn/"+basis_str, molecule=mol_grid, return_wfn=True)
        da_p4 = psi4.core.Matrix.from_array( density_a )
        db_p4 = psi4.core.Matrix.from_array( density_b )
        wfn.V_potential().set_D([ da_p4, db_p4 ])
        wfn.V_potential().properties()[0].set_pointers( da_p4, db_p4 )

        natoms = wfn.nalpha() + wfn.nbeta()
        vpot = wfn.V_potential()
        points = vpot.properties()[0]
        functional = vpot.functional()
        results_grid = data_bucket

        if True:
            mol_dict = wfn.molecule().to_schema(dtype='psi4')
            natoms = len(mol_dict["elem"])
            indx = [i for i in range(natoms) if wfn.molecule().charge(i) != 0.0]
            natoms = len(indx)
            #Atomic numbers and Atomic positions
            zs = [mol_dict["elez"][i] for i in indx]
            rs = [wfn.molecule().geometry().np[i] for i in indx]

        vext_block, vha_block, vxc_a_block, vxc_b_block = [], [], [], []
        vinv_a_block, vinv_b_block = [], []
        xb, yb, zb, zz, vha_z, vxc_az, vxc_bz = [], [], [], [], [], [], []
        vinv_az, vinv_bz = [], []
        xc_e       = 0.0

        for b in range(vpot.nblocks()):

            block = vpot.get_block(b)
            points.compute_points(block)
            npoints = block.npoints()
            lpos = np.array( block.functions_local_to_global() )
            phi = np.array( points.basis_values()["PHI"])[:npoints, :lpos.shape[0]]

            x, y, z, w = np.array(block.x()), np.array(block.y()), np.array(block.z()), np.array(block.w())
            xb.append(x) ; yb.append(y) ; zb.append(z)

            if True: #External
                vext_single = np.zeros(npoints)
                for atom in range(natoms):
                    vext_single += -1.0 * zs[atom] / np.sqrt( (x-rs[atom][0])**2 
                                                            + (y-rs[atom][1])**2
                                                            + (z-rs[atom][2])**2)
                vext_block.append(vext_single)
            if True: #ESP

                grid_block = np.array((x,y,z)).T    
                esp = psi4.core.ESPPropCalc(wfn)
                grid_block = psi4.core.Matrix.from_array(grid_block)
                esp_block = esp.compute_esp_over_grid_in_memory(grid_block).np

                # mol = gto.M(atom='H',
                #             spin=1,
                #             basis=basis)
                # esp_block = eval_vh(mol, grid_block, density_a + density_b )
                vha_block.append(-1.0 * (esp_block + vext_single))

            if True: #RHO / VXC
                if density_a is not None and density_b is None:
                    lD = density_a[(lpos[:, None], lpos)]
                    rho = 2.0 * contract('pm,mn,pn->p', phi, lD, phi) 
                    inp = {}
                    inp["RHO_A"] = psi4.core.Vector.from_array(rho)
                    ret = functional.compute_functional(inp, -1)
                    vk   = np.array(ret["V"])[:npoints]
                    xc_e += np.einsum('a,a', w, vk, optimize=True)

                elif density_a is not None and density_b is not None:
                    lDa  = density_a[(lpos[:, None], lpos)]
                    lDb  = density_b[(lpos[:, None], lpos)]
                    lva  = self.v[:self.naux][lpos]
                    lvb  = self.v[:self.naux][lpos]
                    rho_a = contract('pm,mn,pn->p', phi, lDa, phi)
                    rho_b = contract('pm,mn,pn->p', phi, lDb, phi) 
                    inp = {}
                    inp["RHO_A"] = psi4.core.Vector.from_array(rho_a)
                    inp["RHO_B"] = psi4.core.Vector.from_array(rho_b)
                    ret = functional.compute_functional(inp, -1)
                    vk   = np.array(ret["V"])[:npoints]
                    vxc_a_block.append(np.array(ret["V_RHO_A"])[:npoints])
                    vxc_b_block.append(np.array(ret["V_RHO_B"])[:npoints])
                    xc_e += np.einsum('a,a', w, vk, optimize=True)
            
            # if True: #INVERTED COMPONENT
            #         vinv_a_block.append( contract('pm,m->', phi, lva))
            #         vinv_b_block.append( contract('pm,m->', phi, lvb))

        #Save ordered grid
        x = np.concatenate( [i for i in xb] );  y = np.concatenate( [i for i in yb] ); z = np.concatenate( [i for i in zb] )
        indx = np.argsort(z)
        x,y,z = x[indx], y[indx], z[indx]
        results_grid.x, results_grid.y, results_grid.z = x, y, z
        #Save Exc
        results_grid.exc = float(xc_e) 
        #Save VXC
        vxc_a, vxc_b   = np.concatenate( [i for i in vxc_a_block] ), np.concatenate( [i for i in vxc_b_block] )
        #vinv_a, vinv_b = np.concatenate([i for i in vinv_a_block]), np.concatenate([i for i in vinv_b_block])
        vxc_a, vxc_b   = vxc_a[indx], vxc_b[indx]
        #vinv_a, vinv_b = vinv_a[indx], vinv_b[indx]
        results_grid.vxc_a,  results_grid.vxc_b  = vxc_a, vxc_b
        #results_grid.vinv_a, results_grid.vinv_b = vinv_a, vinv_b
        #Save VHartree
        vha = np.concatenate( [i for i in vha_block] )
        vha = vha[indx]        
        results_grid.vha = vha
        #Save Results along Z axis
        for i in range(len(x)):
            if np.abs(x[i]) < 1e-11:
                if np.abs(y[i]) < 1e-11:
                    zz.append(z[i])
                    vha_z.append(vha[i])
                    vxc_az.append(vxc_a[i])
                    vxc_bz.append(vxc_b[i])
                    # vinv_az.append(vinv_a[i])
                    # vinv_bz.append(vinv_b[i])

        results_grid.zz     = zz
        results_grid.vha_z  = vha_z
        results_grid.vxc_az = vxc_az
        results_grid.vxc_bz = vxc_bz
        # results_grid.vinv_az = vinv_az
        # results_grid.vinv_bz = vinv_bz
 
        self.on_grid = results_grid
 



