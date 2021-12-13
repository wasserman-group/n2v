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
    
    def oucarter(self, maxiter, vxc_grid=None, D_tol=1e-7,
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

        if self.eng_str == 'psi4':
            if vxc_grid is not None:
                plotting_grid = self.eng.grid.grid_to_blocks(vxc_grid)

            # Set pointers for target density
            points_func = self.eng.grid.Vpot.properties()[0]
            if self.ref == 1:
                da_psi4 = psi4.core.Matrix.from_array(self.Dt[0])
                points_func.set_pointers(da_psi4)
            else:
                da_psi4 = psi4.core.Matrix.from_array(self.Dt[0])
                db_psi4 = psi4.core.Matrix.from_array(self.Dt[1])
                points_func.set_pointers(da_psi4, db_psi4)

        # Obtain optimized external Potential
        # Need to get vext_opt_plot and vext_opt_nm

        if self.ref == 1:
            vext_opt_Fock, vext_opt = self.eng.grid.optimized_external_potential()
        else:
            (vext_opt_Fock, vext_opt_Fock_beta), (vext_opt, vext_opt_beta) = self.eng.grid.optimized_external_potential()

        
        vH0_Fock = self.va
        
        Nalpha = self.nalpha
        Nbeta = self.nbeta
        # Initialization.
        if init is None:
            self.Da = np.copy(self.Dt[0])
            self.Coca = np.copy(self.ct[0])
            self.eigvecs_a = self.wfn.epsilon_a().np[:Nalpha]

            self.Db = np.copy(self.Dt[1])
            self.Cocb = np.copy(self.ct[1])
            self.eigvecs_b = self.wfn.epsilon_b().np[:Nbeta]
        elif init.lower()=="continue":
            pass
        else:
            wfn_temp = psi4.energy(init+"/" + self.basis_str, 
                                   molecule=self.mol, 
                                   return_wfn=True)[1]
            self.Da = np.array(wfn_temp.Da())
            self.Coca = np.array(wfn_temp.Ca())[:, :Nalpha]
            self.eigvecs_a = np.array(wfn_temp.epsilon_a())
            self.Db = np.array(wfn_temp.Db())
            self.Cocb = np.array(wfn_temp.Cb())[:, :Nbeta]
            self.eigvecs_b = np.array(wfn_temp.epsilon_b())
            del wfn_temp

        # nerror = self.on_grid_density(Da=self.Dt[0]-self.Da, Db=self.Dt[1]-self.Da, Vpot=self.Vpot)
        # w = self.Vpot.get_np_xyzw()[-1]
        # nerror = np.sum(np.abs(nerror.T) * w)
        # print("nerror", nerror)

        vxc_old = 0.0
        vxc_old_beta = 0.0
        Da_old = 0.0
        eig_old = 0.0
        tauLmP_rho = self._get_l_kinetic_energy_density_directly(self.Dt[0], self.ct[0][:, :Nalpha])
        if self.ref != 1:
            tauLmP_rho_beta = self._get_l_kinetic_energy_density_directly(self.Dt[1], self.ct[1][:, :Nbeta])

        for OC_step in range(1, maxiter+1):
            tauP_rho = self._pauli_kinetic_energy_density(self.Da, self.Coca)
            # _average_local_orbital_energy() taken from mrks.py.
            e_bar = self._average_local_orbital_energy(self.Da, self.Coca, self.eigvecs_a[:Nalpha])
            shift = self.eigvecs_a[Nalpha - 1] - self.wfn.epsilon_a().np[Nalpha - 1]
            # vxc + vext_opt + vH0
            vxc_extH = e_bar - tauLmP_rho - tauP_rho - shift

            if self.ref != 1:
                tauP_rho_beta = self._pauli_kinetic_energy_density(self.Db, self.Cocb)
                e_bar_beta = self._average_local_orbital_energy(self.Db, self.Cocb, self.eigvecs_b[:Nbeta])
                shift_beta = self.eigvecs_b[Nbeta - 1] - self.wfn.epsilon_b().np[Nbeta - 1]
                # vxc + vext_opt + vH0
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

            Vxc_extH_Fock = self.dft_grid_to_fock(vxc_extH, self.Vpot)
            Vxc_Fock = Vxc_extH_Fock - vext_opt_Fock - vH0_Fock

            if self.ref != 1:
                Vxc_extH_Fock_beta = self.dft_grid_to_fock(vxc_extH_beta, self.Vpot)
                Vxc_Fock_beta = Vxc_extH_Fock_beta - vext_opt_Fock_beta - vH0_Fock

            if self.ref == 1:
                self._diagonalize_with_potential_vFock(v=Vxc_Fock)
            else:
                self._diagonalize_with_potential_vFock(v=(Vxc_Fock, Vxc_Fock_beta))


            print(f"Iter: {OC_step}, Density Change: {Derror:2.2e}, Eigenvalue Change: {eerror:2.2e}.")
            # nerror = self.on_grid_density(Da=self.Dt[0] - self.Da, Db=self.Dt[1] - self.Da, Vpot=self.Vpot)
            # nerror = np.sum(np.abs(nerror.T) * w)
            # print("nerror", nerror)

        # Calculate vxc on grid

        vH0 = self.on_grid_esp(grid=grid_info)[1]
        tauLmP_rho = self._get_l_kinetic_energy_density_directly(self.Dt[0], self.ct[0][:, :Nalpha], grid_info=grid_info)
        tauP_rho = self._pauli_kinetic_energy_density(self.Da, self.Coca, grid_info=grid_info)
        shift = self.eigvecs_a[Nalpha - 1] - self.wfn.epsilon_a().np[Nalpha - 1]
        e_bar = self._average_local_orbital_energy(self.Da, self.Coca, self.eigvecs_a[:Nalpha], grid_info=grid_info)
        if self.ref != 1:
            tauLmP_rho_beta = self._get_l_kinetic_energy_density_directly(self.Dt[1], self.ct[1][:, :Nbeta],
                                                                     grid_info=grid_info)
            tauP_rho_beta = self._pauli_kinetic_energy_density(self.Db, self.Cocb, grid_info=grid_info)
            shift_beta = self.eigvecs_b[Nbeta - 1] - self.wfn.epsilon_b().np[Nbeta - 1]
            e_bar_beta = self._average_local_orbital_energy(self.Db, self.Cocb, self.eigvecs_b[:Nbeta], grid_info=grid_info)

        if self.ref == 1:
            self.grid.vxc = e_bar - tauLmP_rho - tauP_rho - vext_opt - vH0 - shift
            return self.grid.vxc, e_bar, tauLmP_rho, tauP_rho, vext_opt, vH0, shift
        else:
            self.grid.vxc = np.array((e_bar - tauLmP_rho - tauP_rho - vext_opt - vH0 - shift,
                                      e_bar_beta - tauLmP_rho_beta - tauP_rho_beta - vext_opt_beta - vH0 - shift_beta
                                      ))
            return self.grid.vxc, (e_bar, e_bar_beta), (tauLmP_rho, tauLmP_rho_beta), \
                   (tauP_rho,tauP_rho_beta), (vext_opt, vext_opt_beta), vH0, (shift, shift_beta)