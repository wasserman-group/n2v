"""
oucarter.py

Functions associated with Ou-Carter inversion
"""

import numpy as np
from opt_einsum import contract
import psi4

class OC():
    """
    Ou-Carter density to potential inversion [1].
    [1] [J. Chem. Theory Comput. 2018, 14, 5680−5689]
    """
    
    def oucarter(self, maxiter, D_tol=1e-7,
             eig_tol=1e-4, frac_old=0.5, init="scan"):
        """
        (23) in [1].
        [1] [J. Chem. Theory Comput. 2018, 14, 5680−5689]

        parameters:
        ----------------------
            maxiter: int
                same as opt_max_iter
            vxc_grid: np.ndarray of shape (3, num_grid_points)
                The final result will be represented on this grid
                default: 1e-4
            D_tol: float, opt
                convergence criteria for density matrices.
                default: 1e-7
            eig_tol: float, opt
                convergence criteria for occupied eigenvalue spectrum.
                default: 1e-4
            frac_old: float, opt
                Linear mixing parameter for current vxc and old vxc.
                If 0, no old vxc is mixed in.
                Should be in [0,1)
                default: 0.5.
            init: string, opt
                Initial guess method.
                default: "SCAN"
                1) If None, input wfn info will be used as initial guess.
                2) If "continue" is given, then it will not initialize
                but use the densities and orbitals stored. Meaningly,
                one can run a quick WY calculation as the initial
                guess. This can also be used to user speficified
                initial guess by setting Da, Coca, eigvec_a.
                3) If it's not continue, it would be expecting a
                method name string that works for psi4. A separate psi4 calculation
                would be performed.

        """
        
        if self.eng_str == 'psi4':
            # if vxc_grid is not None:
            #     plotting_grid = self.eng.grid.grid_to_blocks(vxc_grid)

            # Set pointers for target density
            points_func = self.eng.grid.Vpot.properties()[0]
            if self.ref == 1:
                da_psi4 = psi4.core.Matrix.from_array(self.Dt[0])
                points_func.set_pointers(da_psi4)
            else:
                da_psi4 = psi4.core.Matrix.from_array(self.Dt[0])
                db_psi4 = psi4.core.Matrix.from_array(self.Dt[1])
                points_func.set_pointers(da_psi4, db_psi4)

        
        nalpha     = self.nalpha
        nbeta      = self.nbeta
        vext_tilde = self.eng.grid.external_tilde()          #vext_tilde_nm = ?
        vH0_Fock   = self.va

        # Get initial guess from the target calculation or new one.
        if init is None:
            self.Da = np.copy(self.Dt[0])
            self.Db = np.copy(self.Dt[1])

            self.Ca = np.copy(self.ct[0])
            self.Cb = np.copy(self.ct[1])

            self.eigvecs_a = self.et[0]
            self.eigvecs_b = self.et[1]

        elif init.lower()=="continue":
            pass

        else:
            # Use engine to make a standard calculation
            D, C, e = self.eng.run_single_point(self.eng.mol, self.eng.basis_str, init)
            if self.ref == 1:
                self.Da, self.Db = D/2, D/2
                self.Cocca, self.Coccb = C[:,:self.nalpha], C[:,:self.nbeta] 
                self.Ca, self.Cb = C, C
                self.eigvecs_a, self.eigvecs_b = e, e

            else:
                self.Da, self.Db = D[0], D[1]
                self.Ca, self.Cb = C[0][:,:nalpha], C[1][:,:nbeta]
                self.Cocca, self.Coccb = C[0][:,:nalpha], C[1][:,:nbeta]
                self.eigvecs_a, self.eigvecs_b = e[0], e[1]

        # Ou Carter SCF 
        vxc_eff_old      = 0.0
        vxc_eff_old_beta = 0.0
        Da_old           = np.zeros_like(self.Da)
        eig_old          = np.zeros_like(self.eigvecs_a)

        # Target fixed components
        da0_g      = self.eng.grid.density(density=self.Da)
        gra_da0_g  = self.eng.grid.gradient_density(density=self.Da)
        gra_da0_g  = gra_da0_g[:,0] + gra_da0_g[:,1] + gra_da0_g[:,2]
        lap_da0_g  = self.eng.grid.laplacian_density(density=self.Da)   

        kinetic_energy_method = 'basis'

        # Eq. (26) 1st and 2nd, 5th and 6th. They remain fixed
        vxc_eff0_g    = 0.25 * (lap_da0_g/da0_g) - 0.125 * (np.abs(gra_da0_g)**2/np.abs(da0_g)**2)
        vext_tilde_g  = self.eng.grid.external_tilde(method=kinetic_energy_method)
        vhartree_nm   = self.va

        # Begin SCF iterations
        for iteration in range(1, maxiter+1):
            tau_p   = self.eng.grid.kinetic_energy_density_pauli(self.Cocca, method=kinetic_energy_method)
            e_tilde = self.eng.grid.avg_local_orb_energy(self.Da, self.Cocca, self.eigvecs_a)
            shift   = self.eigvecs_a[nalpha-1] - self.et[0][nalpha-1]

            vxc_eff_g = vxc_eff0_g + e_tilde - tau_p / da0_g - shift

            if self.ref == 2:
                print("Todavia no perra")

        # Linear Mixture 


        # Add exact Hartree + Vext_tilde
            vxc_eff_nm = self.eng.grid.to_ao(  vxc_eff_g - vext_tilde_g  )
            vxc_eff_nm -= vhartree_nm[0]
            
            if self.ref == 2:
                print("Todavia no perra")

            H = self.eng.get_T() + self.eng.get_V() + vxc_eff_nm
            self.Ca, self.Cocca, self.Da, self.eigvecs_a = self.eng.diagonalize( H, nalpha )           

        # Convergence (?) 
            d_error = np.linalg.norm( self.Da - Da_old )
            e_error = np.linalg.norm( self.eigvecs_a - eig_old )

            print(f"Iter: {iteration}, Density Change: {d_error:2.2e}, Eigenvalue Change: {e_error:2.2e} Shift Value:{shift}")

            if ( d_error < D_tol ) and (e_error < eig_tol):
                print("SCF convergence achieved")
                break 

            Da_old  = self.Da.copy()
            eig_old = self.eigvecs_a.copy()

        # Vxc on rectangular grid
        #  Targets. They remain unchanged
        da0_g      = self.eng.grid.density(density=self.Dt[0], grid='rectangular')
        gra_da0_g  = self.eng.grid.gradient_density(density=self.Dt[0], grid='rectangular')
        gra_da0_g  = gra_da0_g[:,0] + gra_da0_g[:,1] + gra_da0_g[:,2]
        lap_da0_g  = self.eng.grid.laplacian_density(density=self.Dt[0], grid='rectangular')
        vext_tilde_g  = self.eng.grid.external_tilde(grid='rectangular', method=kinetic_energy_method)
        vhartree_g   = self.eng.grid.hartree(density=self.Dt[0], grid='rectangular')

        # SCF dependant quantities
        tau_p      = self.eng.grid.kinetic_energy_density_pauli(self.Ca, grid='rectangular', method=kinetic_energy_method)
        e_tilde    = self.eng.grid.avg_local_orb_energy(self.Da, self.Ca, self.eigvecs_a, grid='rectangular')
        vxc_eff0_g    = 0.25 * (lap_da0_g/da0_g) - 0.125 * (np.abs(gra_da0_g)**2/da0_g**2)



        self.vxc0_g     = vxc_eff0_g
        self.vhartree_g = vhartree_g
        self.vext_g     = vext_tilde_g
        self.tau_p      = tau_p/da0_g
        self.e_tilde    = e_tilde

        self.vxc_g = vxc_eff0_g + e_tilde - tau_p/da0_g - vext_tilde_g - vhartree_g



    




        #     # nerror = self.on_grid_density(Da=self.Dt[0] - self.Da, Db=self.Dt[1] - self.Da, Vpot=self.Vpot)
        #     # nerror = np.sum(np.abs(nerror.T) * w)
        #     # print("nerror", nerror)

        # # Calculate vxc on grid

        # vH0 = self.on_grid_esp(grid=grid_info)[1]
        # tauLmP_rho = self._get_l_kinetic_energy_density_directly(self.Dt[0], self.ct[0][:, :Nalpha], grid_info=grid_info)
        # tauP_rho = self._pauli_kinetic_energy_density(self.Da, self.Coca, grid_info=grid_info)
        # shift = self.eigvecs_a[Nalpha - 1] - self.wfn.epsilon_a().np[Nalpha - 1]
        # e_bar = self._average_local_orbital_energy(self.Da, self.Coca, self.eigvecs_a[:Nalpha], grid_info=grid_info)
        # if self.ref != 1:
        #     tauLmP_rho_beta = self._get_l_kinetic_energy_density_directly(self.Dt[1], self.ct[1][:, :Nbeta],
        #                                                              grid_info=grid_info)
        #     tauP_rho_beta = self._pauli_kinetic_energy_density(self.Db, self.Cocb, grid_info=grid_info)
        #     shift_beta = self.eigvecs_b[Nbeta - 1] - self.wfn.epsilon_b().np[Nbeta - 1]
        #     e_bar_beta = self._average_local_orbital_energy(self.Db, self.Cocb, self.eigvecs_b[:Nbeta], grid_info=grid_info)

        # if self.ref == 1:
        #     self.grid.vxc = e_bar - tauLmP_rho - tauP_rho - vext_opt - vH0 - shift
        #     return self.grid.vxc, e_bar, tauLmP_rho, tauP_rho, vext_opt, vH0, shift
        # else:
        #     self.grid.vxc = np.array((e_bar - tauLmP_rho - tauP_rho - vext_opt - vH0 - shift,
        #                               e_bar_beta - tauLmP_rho_beta - tauP_rho_beta - vext_opt_beta - vH0 - shift_beta
        #                               ))
        #     return self.grid.vxc, (e_bar, e_bar_beta), (tauLmP_rho, tauLmP_rho_beta), \
        #            (tauP_rho,tauP_rho_beta), (vext_opt, vext_opt_beta), vH0, (shift, shift_beta)