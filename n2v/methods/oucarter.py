"""
oucarter.py

Functions associated with Ou-Carter inversion
"""

import numpy as np
from opt_einsum import contract
try:
    import psi4
except:
    pass

class OC():
    """
    Ou-Carter density to potential inversion [1].
    [1] [J. Chem. Theory Comput. 2018, 14, 5680−5689]
    """

    def _get_l_kinetic_energy_density_directly(self, D, C, grid_info=None):
        """
        Calculate $\frac{\tau_L^{KS}}{\rho^{KS}}-\frac{\tau_P^{KS}}{\rho^{KS}}$:
        laplace_rho_temp: $\frac{\nabla^2 \rho}{4}$;
        tauW_temp: $\frac{|\napla \rho|^2}{8|\rho|}$;
        tauLmP_rho: $\frac{|\napla \rho|^2}{8|\rho|^2} - \frac{\nabla^2 \rho}{4\rho}$.

        (i.e. the 2dn and 3rd term in eqn. (17) in [1] over $\rho$.):
        """

        if grid_info is None:
            tauLmP_rho = np.zeros(self.Vpot.grid().npoints())
            nblocks = self.Vpot.nblocks()

            points_func = self.Vpot.properties()[0]
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
                l_grid = self.Vpot.get_block(l_block)
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
        wfn_LDA = psi4.energy("SVWN/" + self.basis_str, molecule=self.mol, return_wfn=True)[1]
        Da_LDA = wfn_LDA.Da().np
        Db_LDA = wfn_LDA.Db().np
        Ca_LDA = wfn_LDA.Ca().np
        Cb_LDA = wfn_LDA.Cb().np
        epsilon_a_LDA = wfn_LDA.epsilon_a().np
        epsilon_b_LDA = wfn_LDA.epsilon_b().np
        self.Vpot = wfn_LDA.V_potential()

        vxc_LDA_DFT = self.on_grid_vxc(func_id=1, Da=Da_LDA, Db=Db_LDA, Vpot=self.Vpot)
        vxc_LDA = self.on_grid_vxc(func_id=1,Da=Da_LDA, Db=Db_LDA, grid=grid_info)
        vxc_LDA_DFT += self.on_grid_vxc(func_id=8, Da=Da_LDA, Db=Db_LDA, Vpot=self.Vpot)
        vxc_LDA += self.on_grid_vxc(func_id=8,Da=Da_LDA, Db=Db_LDA, grid=grid_info)
        if self.ref != 1:
            assert vxc_LDA.shape[-1] == 2
            vxc_LDA_beta = vxc_LDA[:,1]
            vxc_LDA = vxc_LDA[:, 0]
            vxc_LDA_DFT_beta = vxc_LDA_DFT[:, 1]
            vxc_LDA_DFT = vxc_LDA_DFT[:, 0]

        # _average_local_orbital_energy() taken from mrks.py.
        e_bar_DFT = self._average_local_orbital_energy(Da_LDA, Ca_LDA[:,:Nalpha], epsilon_a_LDA[:Nalpha])
        e_bar = self._average_local_orbital_energy(Da_LDA, Ca_LDA[:, :Nalpha], epsilon_a_LDA[:Nalpha], grid_info=grid_info)


        tauLmP_rho_DFT = self._get_l_kinetic_energy_density_directly(Da_LDA, Ca_LDA[:,:Nalpha])
        tauLmP_rho = self._get_l_kinetic_energy_density_directly(Da_LDA, Ca_LDA[:,:Nalpha], grid_info=grid_info)

        tauP_rho_DFT = self._pauli_kinetic_energy_density(Da_LDA, Ca_LDA[:,:Nalpha])
        tauP_rho = self._pauli_kinetic_energy_density(Da_LDA, Ca_LDA[:,:Nalpha], grid_info=grid_info)

        tauL_rho_DFT = tauLmP_rho_DFT + tauP_rho_DFT
        tauL_rho = tauLmP_rho + tauP_rho

        vext_opt_no_H_DFT = e_bar_DFT - tauL_rho_DFT - vxc_LDA_DFT
        vext_opt_no_H = e_bar - tauL_rho - vxc_LDA

        J = self.form_jk(Ca_LDA[:,:Nalpha],  Cb_LDA[:,:Nbeta])[0]
        vext_opt_no_H_DFT_Fock = self.dft_grid_to_fock(vext_opt_no_H_DFT, self.Vpot)
        vext_opt_DFT_Fock = vext_opt_no_H_DFT_Fock - J[0] - J[1]

        # # Does vext_opt need a shift?
        # Fock_LDA = self.T + vext_opt_DFT_Fock + J[0] + J[1] + self.dft_grid_to_fock(vxc_LDA_DFT, self.Vpot)
        # eigvecs_a = self.diagonalize(Fock_LDA, self.nalpha)[-1]
        # shift = eigvecs_a[Nalpha-1] - epsilon_a_LDA[Nalpha-1]
        # vext_opt_DFT_Fock -= shift * self.S2
        # print("LDA shift:", shift, eigvecs_a[Nalpha-1], epsilon_a_LDA[Nalpha-1])

        vH = self.on_grid_esp(grid=grid_info, wfn=wfn_LDA)[1]
        vext_opt = vext_opt_no_H - vH
        # vext_opt -= shift

        if self.ref != 1:
            e_bar_DFT_beta = self._average_local_orbital_energy(Db_LDA, Cb_LDA[:,:Nbeta], epsilon_b_LDA[:Nbeta])
            e_bar_beta = self._average_local_orbital_energy(Db_LDA, Cb_LDA[:, :Nbeta], epsilon_b_LDA[:Nbeta], grid_info=grid_info)


            tauLmP_rho_DFT_beta = self._get_l_kinetic_energy_density_directly(Db_LDA, Cb_LDA[:,:Nbeta])
            tauLmP_rho_beta = self._get_l_kinetic_energy_density_directly(Db_LDA, Cb_LDA[:,:Nalpha], grid_info=grid_info)

            tauP_rho_DFT_beta = self._pauli_kinetic_energy_density(Db_LDA, Cb_LDA[:,:Nbeta])
            tauP_rho_beta = self._pauli_kinetic_energy_density(Db_LDA, Cb_LDA[:,:Nbeta], grid_info=grid_info)

            tauL_rho_DFT_beta = tauLmP_rho_DFT_beta + tauP_rho_DFT_beta
            tauL_rho_beta = tauLmP_rho_beta + tauP_rho_beta

            vext_opt_no_H_DFT_beta = e_bar_DFT_beta - tauL_rho_DFT_beta - vxc_LDA_DFT_beta
            vext_opt_no_H_beta = e_bar_beta - tauL_rho_beta - vxc_LDA_beta

            vext_opt_no_H_DFT_Fock_beta = self.dft_grid_to_fock(vext_opt_no_H_DFT_beta, self.Vpot)
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
        
        grid_info = self.grid_to_blocks(vxc_grid)
        if self.ref == 1:
            grid_info[-1].set_pointers(self.wfn.Da())
        else:
            grid_info[-1].set_pointers(self.wfn.Da(), self.wfn.Db())

        if self.ref == 1:
            vext_opt_Fock, vext_opt = self._get_optimized_external_potential(grid_info)
        else:
            (vext_opt_Fock, vext_opt_Fock_beta), (vext_opt, vext_opt_beta) = self._get_optimized_external_potential(grid_info)

        assert self.Vpot is not None
        
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