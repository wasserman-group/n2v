"""
oucarter.py

Functions associated with Ou-Carter inversion
"""

import numpy as np
import psi4

class OC():
    """
    Ou-Carter density to potential inversion [1].
    [1] [J. Chem. Theory Comput. 2018, 14, 5680−5689]
    """

    def _get_optimized_external_potential(self, grid_info, average_alpha_beta=False):
        """
        $
        v^{~}{ext}(r) = \epsilon^{-LDA}(r)
        - \frac{\tau^{LDA}{L}}{n^{LDA}(r)}
        - v_{H}^{LDA}(r) - v_{xc}^{LDA}(r)
        $
        (22) in [1].
        """

        Nalpha = self.nalpha
        Nbeta = self.nbeta

        # SVWN calculation
        wfn_LDA = psi4.energy("SVWN/" + self.eng.basis_str, molecule=self.eng.mol, return_wfn=True)[1]
        Da_LDA = wfn_LDA.Da().np
        Db_LDA = wfn_LDA.Db().np
        Ca_LDA = wfn_LDA.Ca().np
        Cb_LDA = wfn_LDA.Cb().np
        epsilon_a_LDA = wfn_LDA.epsilon_a().np
        epsilon_b_LDA = wfn_LDA.epsilon_b().np
        LDA_Vpot = wfn_LDA.V_potential()
        self.LDA_Vpot = LDA_Vpot

        vxc_LDA_DFT  = self.eng.grid.vxc(func_id=1, Da=Da_LDA, Db=Db_LDA, Vpot=LDA_Vpot) 
        vxc_LDA_DFT += self.eng.grid.vxc(func_id=8, Da=Da_LDA, Db=Db_LDA, Vpot=LDA_Vpot)
        vxc_LDA      = self.eng.grid.vxc(func_id=1,Da=Da_LDA, Db=Db_LDA, grid=grid_info) 
        vxc_LDA     += self.eng.grid.vxc(func_id=8,Da=Da_LDA, Db=Db_LDA, grid=grid_info)
        if self.ref != 1:
            assert vxc_LDA.shape[-1] == 2
            vxc_LDA_beta = vxc_LDA[:,1]
            vxc_LDA = vxc_LDA[:, 0]
            vxc_LDA_DFT_beta = vxc_LDA_DFT[:, 1]
            vxc_LDA_DFT = vxc_LDA_DFT[:, 0]

        # _average_local_orbital_energy() taken from mrks.py.
        e_bar_DFT      = self.eng.grid._average_local_orbital_energy(Da_LDA, Ca_LDA[:,:Nalpha], epsilon_a_LDA[:Nalpha], Vpot=LDA_Vpot)
        tauLmP_rho_DFT = self.eng.grid._get_l_kinetic_energy_density_directly(Da_LDA, Ca_LDA[:,:Nalpha], Vpot=LDA_Vpot)
        tauP_rho_DFT   = self.eng.grid._pauli_kinetic_energy_density(Da_LDA, Ca_LDA[:,:Nalpha], Vpot=LDA_Vpot)

        tauLmP_rho     = self.eng.grid._get_l_kinetic_energy_density_directly(Da_LDA, Ca_LDA[:,:Nalpha], grid_info=grid_info)
        e_bar          = self.eng.grid._average_local_orbital_energy(Da_LDA, Ca_LDA[:, :Nalpha], epsilon_a_LDA[:Nalpha], grid_info=grid_info)
        tauP_rho       = self.eng.grid._pauli_kinetic_energy_density(Da_LDA, Ca_LDA[:,:Nalpha], grid_info=grid_info)
        # print("taulp", tauLmP_rho)
        # print("Ebar", e_bar)
        # print("taup", tauP_rho)

        tauL_rho_DFT = tauLmP_rho_DFT + tauP_rho_DFT
        tauL_rho = tauLmP_rho + tauP_rho

        vext_opt_no_H_DFT = e_bar_DFT - tauL_rho_DFT - vxc_LDA_DFT
        vext_opt_no_H = e_bar - tauL_rho - vxc_LDA

        J = self.compute_hartree(Ca_LDA[:,:Nalpha],  Cb_LDA[:,:Nbeta])
        vext_opt_no_H_DFT_Fock = self.eng.grid.dft_grid_to_fock(vext_opt_no_H_DFT, LDA_Vpot)
        vext_opt_DFT_Fock = vext_opt_no_H_DFT_Fock - J[0] - J[1]

        # # Does vext_opt need a shift?
        # Fock_LDA = self.T + vext_opt_DFT_Fock + J[0] + J[1] + self.dft_grid_to_fock(vxc_LDA_DFT, self.Vpot)
        # eigvecs_a = self.diagonalize(Fock_LDA, self.nalpha)[-1]
        # shift = eigvecs_a[Nalpha-1] - epsilon_a_LDA[Nalpha-1]
        # vext_opt_DFT_Fock -= shift * self.S2
        # print("LDA shift:", shift, eigvecs_a[Nalpha-1], epsilon_a_LDA[Nalpha-1])

        vH = self.eng.grid.esp(grid=grid_info, wfn=wfn_LDA)[1]
        vext_opt = vext_opt_no_H - vH
        # vext_opt -= shift

        if self.ref != 1:
            e_bar_DFT_beta = self.eng.grid._average_local_orbital_energy(Db_LDA, Cb_LDA[:,:Nbeta], epsilon_b_LDA[:Nbeta], Vpot=LDA_Vpot)
            e_bar_beta = self.eng.grid._average_local_orbital_energy(Db_LDA, Cb_LDA[:, :Nbeta], epsilon_b_LDA[:Nbeta], grid_info=grid_info)

            tauLmP_rho_DFT_beta = self.eng.grid._get_l_kinetic_energy_density_directly(Db_LDA, Cb_LDA[:,:Nbeta], Vpot=LDA_Vpot)
            tauLmP_rho_beta = self.eng.grid._get_l_kinetic_energy_density_directly(Db_LDA, Cb_LDA[:,:Nalpha], grid_info=grid_info)

            tauP_rho_DFT_beta = self.eng.grid._pauli_kinetic_energy_density(Db_LDA, Cb_LDA[:,:Nbeta], Vpot=LDA_Vpot)
            tauP_rho_beta = self.eng.grid._pauli_kinetic_energy_density(Db_LDA, Cb_LDA[:,:Nbeta], grid_info=grid_info)

            tauL_rho_DFT_beta = tauLmP_rho_DFT_beta + tauP_rho_DFT_beta
            tauL_rho_beta = tauLmP_rho_beta + tauP_rho_beta

            vext_opt_no_H_DFT_beta = e_bar_DFT_beta - tauL_rho_DFT_beta - vxc_LDA_DFT_beta
            vext_opt_no_H_beta = e_bar_beta - tauL_rho_beta - vxc_LDA_beta

            vext_opt_no_H_DFT_Fock_beta = self.eng.grid.dft_grid_to_fock(vext_opt_no_H_DFT_beta, LDA_Vpot)
            vext_opt_DFT_Fock_beta = vext_opt_no_H_DFT_Fock_beta - J[0] - J[1]

            vext_opt_beta = vext_opt_no_H_beta - vH

            # vext_opt_DFT_Fock = (vext_opt_DFT_Fock + vext_opt_DFT_Fock_beta) * 0.5
            # vext_opt = (vext_opt + vext_opt_beta) * 0.5

            return (vext_opt_DFT_Fock, vext_opt_DFT_Fock_beta), (vext_opt, vext_opt_beta)
        return vext_opt_DFT_Fock, vext_opt
    
    def oucarter(self, maxiter, vxc_grid, D_tol=1e-7,
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
        
        # if self.ref != 1:
        #     raise ValueError("Currently only supports Spin-Restricted "
        #                      "calculations since Spin-Unrestricted CI "
        #                      "is not supported by Psi4.")

        grid_info = self.eng.grid.grid_to_blocks(vxc_grid) 
        if self.ref == 1:
            grid_info[-1].set_pointers(self.eng.wfn.Da())
            vext_opt_Fock, vext_opt = self._get_optimized_external_potential(grid_info)
        else:
            grid_info[-1].set_pointers(self.eng.wfn.Da(), self.eng.wfn.Db())
            (vext_opt_Fock, vext_opt_Fock_beta), (vext_opt, vext_opt_beta) = self._get_optimized_external_potential(grid_info)

        # Print Hasta aca vamos bien

        assert self.LDA_Vpot is not None        
        vH0_Fock = self.va

        Nalpha = self.nalpha
        Nbeta = self.nbeta
        # Initialization.
        if init is None:
            self.Da        = np.copy(self.Dt[0])
            self.Db        = np.copy(self.Dt[1])
            self.Coca      = np.copy(self.ct[0])
            self.Cocb      = np.copy(self.ct[1])
            self.eigvecs_a = self.eng.wfn.epsilon_a().np[:Nalpha]
            self.eigvecs_b = self.eng.wfn.epsilon_b().np[:Nbeta]
        elif init.lower()=="continue":
            pass
        else:
            wfn_temp = psi4.energy(init+"/" + self.eng.grid.basis_str, 
                                   molecule=self.eng.grid.mol, 
                                   return_wfn=True)[1]
            self.Da = np.array(wfn_temp.Da())
            self.Db = np.array(wfn_temp.Db())
            self.Coca = np.array(wfn_temp.Ca())[:, :Nalpha]
            self.Cocb = np.array(wfn_temp.Cb())[:, :Nbeta]
            self.eigvecs_a = np.array(wfn_temp.epsilon_a())
            self.eigvecs_b = np.array(wfn_temp.epsilon_b())
            del wfn_temp

        # nerror = self.on_grid_density(Da=self.Dt[0]-self.Da, Db=self.Dt[1]-self.Da, Vpot=self.Vpot)
        # w = self.Vpot.get_np_xyzw()[-1]
        # nerror = np.sum(np.abs(nerror.T) * w)
        # print("nerror", nerror)

        vxc_old      = 0.0
        vxc_old_beta = 0.0
        Da_old       = 0.0
        eig_old      = 0.0
        tauLmP_rho = self.eng.grid._get_l_kinetic_energy_density_directly(self.Dt[0], self.ct[0][:, :Nalpha])
        if self.ref != 1:
            tauLmP_rho_beta = self.eng.grid._get_l_kinetic_energy_density_directly(self.Dt[1], self.ct[1][:, :Nbeta])

        for OC_step in range(1, maxiter+1):
            tauP_rho = self.eng.grid._pauli_kinetic_energy_density(self.Da, self.Coca)
            e_bar    = self.eng.grid._average_local_orbital_energy(self.Da, self.Coca, self.eigvecs_a[:Nalpha])
            shift    = self.eigvecs_a[Nalpha - 1] - self.eng.wfn.epsilon_a().np[Nalpha - 1]
            vxc_extH = e_bar - tauLmP_rho - tauP_rho - shift

            if self.ref != 1:
                tauP_rho_beta = self.eng.grid._pauli_kinetic_energy_density(self.Db, self.Cocb)
                e_bar_beta    = self.eng.grid._average_local_orbital_energy(self.Db, self.Cocb, self.eigvecs_b[:Nbeta])
                shift_beta    = self.eigvecs_b[Nbeta - 1] - self.eng.wfn.epsilon_b().np[Nbeta - 1]
                vxc_extH_beta = e_bar_beta - tauLmP_rho_beta - tauP_rho_beta - shift_beta

            Derror = np.linalg.norm(self.Da - Da_old) / self.nbf ** 2
            eerror = (np.linalg.norm(self.eigvecs_a[:Nalpha] - eig_old) / Nalpha)
            if (Derror < D_tol) and (eerror < eig_tol):
                print("KSDFT stops updating.")
                break

            # linear Mixture
            if OC_step != 1:
                vxc_extH = vxc_extH * (1 - frac_old) + vxc_old * frac_old
                if self.ref != 1:
                    vxc_extH_beta = vxc_extH_beta * (1 - frac_old) + vxc_old_beta * frac_old

            vxc_old = np.copy(vxc_extH)
            if self.ref != 1:
                vxc_old_beta = np.copy(vxc_extH_beta)

            # Save old data.
            Da_old = np.copy(self.Da)
            eig_old = np.copy(self.eigvecs_a[:Nalpha])

            Vxc_extH_Fock = self.eng.grid.dft_grid_to_fock(vxc_extH, self.LDA_Vpot)
            Vxc_Fock      = Vxc_extH_Fock - vext_opt_Fock - vH0_Fock

            # print("1", np.diag(Vxc_extH_Fock))
            # print("2", np.diag(vext_opt_Fock))
            # print("3", np.diag(vH0_Fock))
            # print("4", np.diag(Vxc_Fock))

            if self.ref != 1:
                Vxc_extH_Fock_beta = self.eng.grid.dft_grid_to_fock(vxc_extH_beta, self.LDA_Vpot)
                Vxc_Fock_beta      = Vxc_extH_Fock_beta - vext_opt_Fock_beta - vH0_Fock

            if self.ref == 1:
                self._diagonalize_with_potential_vFock(v=Vxc_Fock)
            else:
                self._diagonalize_with_potential_vFock(v=(Vxc_Fock, Vxc_Fock_beta))

            print(f"Iter: {OC_step}, Density Change: {Derror:2.2e}, Eigenvalue Change: {eerror:2.2e}.")
            # nerror = self.on_grid_density(Da=self.Dt[0] - self.Da, Db=self.Dt[1] - self.Da, Vpot=self.Vpot)
            # nerror = np.sum(np.abs(nerror.T) * w)
            # print("nerror", nerror)

        # Calculate vxc on grid
        vH0        = self.eng.grid.esp(grid=grid_info, wfn=self.eng.wfn)[1]
        tauLmP_rho = self.eng.grid._get_l_kinetic_energy_density_directly(self.Dt[0], self.ct[0][:, :Nalpha], grid_info=grid_info)
        tauP_rho   = self.eng.grid._pauli_kinetic_energy_density(self.Da, self.Coca, grid_info=grid_info)
        e_bar      = self.eng.grid._average_local_orbital_energy(self.Da, self.Coca, self.eigvecs_a[:Nalpha], grid_info=grid_info)
        shift      = self.eigvecs_a[Nalpha - 1] - self.eng.wfn.epsilon_a().np[Nalpha - 1]

        if self.ref != 1:
            tauLmP_rho_beta = self.eng.grid._get_l_kinetic_energy_density_directly(self.Dt[1], self.ct[1][:, :Nbeta],grid_info=grid_info)
            tauP_rho_beta   = self.eng.grid._pauli_kinetic_energy_density(self.Db, self.Cocb, grid_info=grid_info)
            e_bar_beta      = self.eng.grid._average_local_orbital_energy(self.Db, self.Cocb, self.eigvecs_b[:Nbeta], grid_info=grid_info)
            shift_beta      = self.eigvecs_b[Nbeta - 1] - self.eng.wfn.epsilon_b().np[Nbeta - 1]

        if self.ref == 1:
            self.grid_vxc = e_bar - tauLmP_rho - tauP_rho - vext_opt - vH0 - shift
            return self.grid_vxc, e_bar, tauLmP_rho, tauP_rho, vext_opt, vH0, shift
        else:
            self.grid_vxc = np.array((e_bar - tauLmP_rho - tauP_rho - vext_opt - vH0 - shift,
                                      e_bar_beta - tauLmP_rho_beta - tauP_rho_beta - vext_opt_beta - vH0 - shift_beta
                                      ))
            return self.grid_vxc, (e_bar, e_bar_beta), (tauLmP_rho, tauLmP_rho_beta), \
                   (tauP_rho,tauP_rho_beta), (vext_opt, vext_opt_beta), vH0, (shift, shift_beta)
# """
# oucarter.py

# Functions associated with Ou-Carter inversion
# """

# import numpy as np
# from opt_einsum import contract
# import psi4

# class OC():
#     """
#     Ou-Carter density to potential inversion [1].
#     [1] [J. Chem. Theory Comput. 2018, 14, 5680−5689]
#     """
    
#     def oucarter(self, maxiter, D_tol=1e-7,
#              eig_tol=1e-4, frac_old=0.5, init="scan"):
#         """
#         (23) in [1].
#         [1] [J. Chem. Theory Comput. 2018, 14, 5680−5689]

#         parameters:
#         ----------------------
#             maxiter: int
#                 same as opt_max_iter
#             vxc_grid: np.ndarray of shape (3, num_grid_points)
#                 The final result will be represented on this grid
#                 default: 1e-4
#             D_tol: float, opt
#                 convergence criteria for density matrices.
#                 default: 1e-7
#             eig_tol: float, opt
#                 convergence criteria for occupied eigenvalue spectrum.
#                 default: 1e-4
#             frac_old: float, opt
#                 Linear mixing parameter for current vxc and old vxc.
#                 If 0, no old vxc is mixed in.
#                 Should be in [0,1)
#                 default: 0.5.
#             init: string, opt
#                 Initial guess method.
#                 default: "SCAN"
#                 1) If None, input wfn info will be used as initial guess.
#                 2) If "continue" is given, then it will not initialize
#                 but use the densities and orbitals stored. Meaningly,
#                 one can run a quick WY calculation as the initial
#                 guess. This can also be used to user speficified
#                 initial guess by setting Da, Coca, eigvec_a.
#                 3) If it's not continue, it would be expecting a
#                 method name string that works for psi4. A separate psi4 calculation
#                 would be performed.

#         """
        
#         if self.eng_str == 'psi4':
#             # if vxc_grid is not None:
#             #     plotting_grid = self.eng.grid.grid_to_blocks(vxc_grid)

#             # Set pointers for target density
#             points_func = self.eng.grid.Vpot.properties()[0]
#             if self.ref == 1:
#                 da_psi4 = psi4.core.Matrix.from_array(self.Dt[0])
#                 points_func.set_pointers(da_psi4)
#             else:
#                 da_psi4 = psi4.core.Matrix.from_array(self.Dt[0])
#                 db_psi4 = psi4.core.Matrix.from_array(self.Dt[1])
#                 points_func.set_pointers(da_psi4, db_psi4)

        
#         nalpha     = self.nalpha
#         nbeta      = self.nbeta
#         vext_tilde = self.eng.grid.external_tilde()          #vext_tilde_nm = ?
#         vH0_Fock   = self.va

#         # Get initial guess from the target calculation or new one.
#         if init is None:
#             self.Da = np.copy(self.Dt[0])
#             self.Db = np.copy(self.Dt[1])

#             self.Ca = np.copy(self.ct[0])
#             self.Cb = np.copy(self.ct[1])

#             self.eigvecs_a = self.et[0]
#             self.eigvecs_b = self.et[1]

#         elif init.lower()=="continue":
#             pass

#         else:
#             # Use engine to make a standard calculation
#             D, C, e = self.eng.run_single_point(self.eng.mol, self.eng.basis_str, init)
#             if self.ref == 1:
#                 self.Da, self.Db = D/2, D/2
#                 self.Cocca, self.Coccb = C[:,:self.nalpha], C[:,:self.nbeta] 
#                 self.Ca, self.Cb = C, C
#                 self.eigvecs_a, self.eigvecs_b = e, e

#             else:
#                 self.Da, self.Db = D[0], D[1]
#                 self.Ca, self.Cb = C[0][:,:nalpha], C[1][:,:nbeta]
#                 self.Cocca, self.Coccb = C[0][:,:nalpha], C[1][:,:nbeta]
#                 self.eigvecs_a, self.eigvecs_b = e[0], e[1]

#         # Ou Carter SCF 
#         vxc_eff_old      = 0.0
#         vxc_eff_old_beta = 0.0
#         Da_old           = np.zeros_like(self.Da)
#         eig_old          = np.zeros_like(self.eigvecs_a)

#         # Target fixed components
#         da0_g      = self.eng.grid.density(density=self.Da)
#         gra_da0_g  = self.eng.grid.gradient_density(density=self.Da)
#         gra_da0_g  = gra_da0_g[:,0] + gra_da0_g[:,1] + gra_da0_g[:,2]
#         lap_da0_g  = self.eng.grid.laplacian_density(density=self.Da)   

#         kinetic_energy_method = 'basis'

#         # Eq. (26) 1st and 2nd, 5th and 6th. They remain fixed
#         vxc_eff0_g    = 0.25 * (lap_da0_g/da0_g) - 0.125 * (np.abs(gra_da0_g)**2/np.abs(da0_g)**2)
#         vext_tilde_g  = self.eng.grid.external_tilde(method=kinetic_energy_method)
#         vhartree_nm   = self.va

#         # Begin SCF iterations
#         for iteration in range(1, maxiter+1):
#             tau_p   = self.eng.grid.kinetic_energy_density_pauli(self.Cocca, method=kinetic_energy_method)
#             e_tilde = self.eng.grid.avg_local_orb_energy(self.Da, self.Cocca, self.eigvecs_a)
#             shift   = self.eigvecs_a[nalpha-1] - self.et[0][nalpha-1]

#             vxc_eff_g = vxc_eff0_g + e_tilde - tau_p / da0_g - shift

#             if self.ref == 2:
#                 print("Todavia no perra")

#         # Linear Mixture 


#         # Add exact Hartree + Vext_tilde
#             vxc_eff_nm = self.eng.grid.to_ao(  vxc_eff_g - vext_tilde_g  )
#             vxc_eff_nm -= vhartree_nm[0]
            
#             if self.ref == 2:
#                 print("Todavia no perra")

#             H = self.eng.get_T() + self.eng.get_V() + vxc_eff_nm
#             self.Ca, self.Cocca, self.Da, self.eigvecs_a = self.eng.diagonalize( H, nalpha )           

#         # Convergence (?) 
#             d_error = np.linalg.norm( self.Da - Da_old )
#             e_error = np.linalg.norm( self.eigvecs_a - eig_old )

#             print(f"Iter: {iteration}, Density Change: {d_error:2.2e}, Eigenvalue Change: {e_error:2.2e} Shift Value:{shift}")

#             if ( d_error < D_tol ) and (e_error < eig_tol):
#                 print("SCF convergence achieved")
#                 break 

#             Da_old  = self.Da.copy()
#             eig_old = self.eigvecs_a.copy()

#         # Vxc on rectangular grid
#         #  Targets. They remain unchanged
#         da0_g      = self.eng.grid.density(density=self.Dt[0], grid='rectangular')
#         gra_da0_g  = self.eng.grid.gradient_density(density=self.Dt[0], grid='rectangular')
#         gra_da0_g  = gra_da0_g[:,0] + gra_da0_g[:,1] + gra_da0_g[:,2]
#         lap_da0_g  = self.eng.grid.laplacian_density(density=self.Dt[0], grid='rectangular')
#         vext_tilde_g  = self.eng.grid.external_tilde(grid='rectangular', method=kinetic_energy_method)
#         vhartree_g   = self.eng.grid.hartree(density=self.Dt[0], grid='rectangular')

#         # SCF dependant quantities
#         tau_p      = self.eng.grid.kinetic_energy_density_pauli(self.Ca, grid='rectangular', method=kinetic_energy_method)
#         e_tilde    = self.eng.grid.avg_local_orb_energy(self.Da, self.Ca, self.eigvecs_a, grid='rectangular')
#         vxc_eff0_g    = 0.25 * (lap_da0_g/da0_g) - 0.125 * (np.abs(gra_da0_g)**2/da0_g**2)



#         self.vxc0_g     = vxc_eff0_g
#         self.vhartree_g = vhartree_g
#         self.vext_g     = vext_tilde_g
#         self.tau_p      = tau_p/da0_g
#         self.e_tilde    = e_tilde

#         self.vxc_g = vxc_eff0_g + e_tilde - tau_p/da0_g - vext_tilde_g - vhartree_g



    




#         #     # nerror = self.on_grid_density(Da=self.Dt[0] - self.Da, Db=self.Dt[1] - self.Da, Vpot=self.Vpot)
#         #     # nerror = np.sum(np.abs(nerror.T) * w)
#         #     # print("nerror", nerror)

#         # # Calculate vxc on grid

#         # vH0 = self.on_grid_esp(grid=grid_info)[1]
#         # tauLmP_rho = self._get_l_kinetic_energy_density_directly(self.Dt[0], self.ct[0][:, :Nalpha], grid_info=grid_info)
#         # tauP_rho = self._pauli_kinetic_energy_density(self.Da, self.Coca, grid_info=grid_info)
#         # shift = self.eigvecs_a[Nalpha - 1] - self.wfn.epsilon_a().np[Nalpha - 1]
#         # e_bar = self._average_local_orbital_energy(self.Da, self.Coca, self.eigvecs_a[:Nalpha], grid_info=grid_info)
#         # if self.ref != 1:
#         #     tauLmP_rho_beta = self._get_l_kinetic_energy_density_directly(self.Dt[1], self.ct[1][:, :Nbeta],
#         #                                                              grid_info=grid_info)
#         #     tauP_rho_beta = self._pauli_kinetic_energy_density(self.Db, self.Cocb, grid_info=grid_info)
#         #     shift_beta = self.eigvecs_b[Nbeta - 1] - self.wfn.epsilon_b().np[Nbeta - 1]
#         #     e_bar_beta = self._average_local_orbital_energy(self.Db, self.Cocb, self.eigvecs_b[:Nbeta], grid_info=grid_info)

#         # if self.ref == 1:
#         #     self.grid.vxc = e_bar - tauLmP_rho - tauP_rho - vext_opt - vH0 - shift
#         #     return self.grid.vxc, e_bar, tauLmP_rho, tauP_rho, vext_opt, vH0, shift
#         # else:
#         #     self.grid.vxc = np.array((e_bar - tauLmP_rho - tauP_rho - vext_opt - vH0 - shift,
#         #                               e_bar_beta - tauLmP_rho_beta - tauP_rho_beta - vext_opt_beta - vH0 - shift_beta
#         #                               ))
#         #     return self.grid.vxc, (e_bar, e_bar_beta), (tauLmP_rho, tauLmP_rho_beta), \
#         #            (tauP_rho,tauP_rho_beta), (vext_opt, vext_opt_beta), vH0, (shift, shift_beta)