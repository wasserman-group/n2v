"""
grider.py
"""

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


    def density(self, Da, Db, grid=None):
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

    def get_orbitals():
        pass
