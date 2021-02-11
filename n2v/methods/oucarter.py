"""
oucarter.py

Functions associated with Ou-Carter inversion
"""

import numpy as np
from opt_einsum import contract
import psi4
import time

class OC():
    """
    Ou-Carter density to potential inversion [1].
    [1] [J. Chem. Theory Comput. 2018, 14, 5680−5689]
    """

    def _get_l_kinetic_energy_density_directly(self, D, grid_info=None):
        """
        Calculate $\frac{\tau_L^{KS}}{\rho^{KS}}-\frac{\tau_P^{KS}}{\rho^{KS}}$,
        (i.e. the 2dn and 3rd term in eqn. (17) in [1]):
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
            l_phi = np.array(points_func.basis_values()["PHI"])[:l_npoints, :l_lpos.shape[0]]
            l_phi_x = np.array(points_func.basis_values()["PHI_X"])[:l_npoints, :l_lpos.shape[0]]
            l_phi_y = np.array(points_func.basis_values()["PHI_Y"])[:l_npoints, :l_lpos.shape[0]]
            l_phi_z = np.array(points_func.basis_values()["PHI_Z"])[:l_npoints, :l_lpos.shape[0]]
            l_phi_xx = np.array(points_func.basis_values()["PHI_XX"])[:l_npoints, :l_lpos.shape[0]]
            l_phi_yy = np.array(points_func.basis_values()["PHI_YY"])[:l_npoints, :l_lpos.shape[0]]
            l_phi_zz = np.array(points_func.basis_values()["PHI_ZZ"])[:l_npoints, :l_lpos.shape[0]]

            lD = D[(l_lpos[:, None], l_lpos)]

            rho = contract('pm,mn,pn->p', l_phi, lD, l_phi)
            rho_inv = 1/rho
            # rho_KS = contract('pm,mn,pn->p', l_phi, lD_KS, l_phi)
            #
            # # Calculate slice of \frac{\tau_p^KS}{\rho^KS}
            # part_x = contract('pm,mi,nj,pn->ijp', l_phi, lC_KS, lC_KS, l_phi_x)
            # part_y = contract('pm,mi,nj,pn->ijp', l_phi, lC_KS, lC_KS, l_phi_y)
            # part_z = contract('pm,mi,nj,pn->ijp', l_phi, lC_KS, lC_KS, l_phi_z)
            # part1_x = (part_x - np.transpose(part_x, (1, 0, 2))) ** 2
            # part1_y = (part_y - np.transpose(part_y, (1, 0, 2))) ** 2
            # part1_z = (part_z - np.transpose(part_z, (1, 0, 2))) ** 2
            # taup_rho_temp = np.sum((part1_x + part1_y + part1_z).T, axis=(1,2)) / rho ** 2 * (0.5 * 0.5)


            # Calculate the second term 0.25*\nabla^2\rho
            laplace_rho_temp = contract('ab,pa,pb->p', lD, l_phi, l_phi_xx + l_phi_yy + l_phi_zz)
            laplace_rho_temp += contract('pm, mn, pn->p', l_phi_x,lD, l_phi_x)
            laplace_rho_temp += contract('pm, mn, pn->p', l_phi_y,lD, l_phi_y)
            laplace_rho_temp += contract('pm, mn, pn->p', l_phi_z,lD, l_phi_z)
            laplace_rho_temp *= 0.25 * 2

            # Calculate the third term |nabla rho|^2 / 8
            tauW_temp = 4 * contract('pm, mn, pn->p', l_phi, lD, l_phi_x) ** 2
            tauW_temp += 4 * contract('pm, mn, pn->p', l_phi, lD, l_phi_y) ** 2
            tauW_temp += 4 * contract('pm, mn, pn->p', l_phi, lD, l_phi_z) ** 2
            tauW_temp *= rho_inv * 0.125

            tauLmP_rho[iw: iw + l_npoints] = (-laplace_rho_temp + tauW_temp) * rho_inv
            iw += l_npoints
        assert iw == tauLmP_rho.shape[0], "Somehow the whole space is not fully integrated."
        return tauLmP_rho

    def _get_optimized_external_potential(self, grid_info=None):
        """
        (22) in [1].

        return:
            if grid_info is None:
                returns vxc Fock matrix: np.ndarray of shape (nbf, nbf)
            else:
                return vxc on the given grid: np.ndarray of shape (num_grid_points,)
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

        if grid_info is None:
            vxc_LDA = self.on_grid_vxc(Da=Da_LDA, Db=Db_LDA, vpot=self.Vpot)
        else:
            vxc_LDA = self.on_grid_vxc(Da=Da_LDA, Db=Db_LDA, grid=grid_info)

        e_bar = self._average_local_orbital_energy(Da_LDA, Ca_LDA[:,:Nalpha],
                                                   epsilon_a_LDA[:Nalpha], grid_info=grid_info)
        tauLmP_rho = self._get_l_kinetic_energy_density_directly(Da_LDA, grid_info=grid_info)
        tauP_rho = self._pauli_kinetic_energy_density(Da_LDA, Ca_LDA[:,:Nalpha], grid_info=grid_info)
        tauL_rho = tauLmP_rho + tauP_rho
        vext_opt_no_H = e_bar - tauL_rho - vxc_LDA

        if grid_info is None:
            vext_opt_no_H_Fock = self.dft_grid_to_fock(vext_opt_no_H, self.Vpot)

            J, _ = self.form_jk(psi4.core.Matrix.from_array(Ca_LDA[:,:Nalpha]),
                                psi4.core.Matrix.from_array(Cb_LDA[:,:Nbeta]))

            vH_Fock = J[0] + J[1]

            vext_opt = vext_opt_no_H_Fock - vH_Fock
            return vext_opt

        else:
            vH = self.on_grid_esp(grid=grid_info, wfn=wfn_LDA)[1]
            return vext_opt_no_H - vH
    
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
        
        if self.ref != 1:
            raise ValueError("Currently only supports Spin-Restricted "
                             "calculations since Spin-Unrestricted CI "
                             "is not supported by Psi4.")

        
        if self.guide_potential_components[0] != "hartree":
            raise ValueError("Hartree potential is necessary as the guide potential.")
        
        
        vext_opt_Fock = self._get_optimized_external_potential()
        assert self.Vpot is not None
        
        vH0_Fock = self.va
        
        Nalpha = self.nalpha
        # Initialization.
        if init is None:
            self.Da = np.copy(self.nt[0])
            self.Coca = np.copy(self.ct[0])
            self.eigvecs_a = self.wfn.epsilon_a().np[:Nalpha]
        elif init.lower()=="continue":
            pass
        else:
            wfn_temp = psi4.energy(init+"/" + self.basis_str, 
                                   molecule=self.mol, 
                                   return_wfn=True)[1]
            self.Da = np.array(wfn_temp.Da())
            self.Coca = np.array(wfn_temp.Ca())[:, :Nalpha]
            self.eigvecs_a = np.array(wfn_temp.epsilon_a())
            del wfn_temp

        vxc_old = 0.0
        Da_old = 0.0
        eig_old = 0.0
        for OC_step in range(1, maxiter+1):
            tauLmP_rho = self._get_l_kinetic_energy_density_directly(self.nt[0])
            tauP_rho = self._pauli_kinetic_energy_density(self.Da, self.Coca)
            e_bar = self._average_local_orbital_energy(self.Da, self.Coca, self.eigvecs_a[:Nalpha])

            # vxc + vext_opt + vH0
            vxc_extH = e_bar - tauLmP_rho - tauP_rho

            Derror = np.linalg.norm(self.Da - Da_old) / self.nbf ** 2
            eerror = (np.linalg.norm(self.eigvecs_a[:Nalpha] - eig_old) / Nalpha)
            if (Derror < D_tol) and (eerror < eig_tol):
                print("KSDFT stops updating.")
                break

            # linear Mixture
            if OC_step != 1:
                vxc_extH = vxc_extH * (1 - frac_old) + vxc_old * frac_old

            # Save old data.
            vxc_old = np.copy(vxc_extH)
            Da_old = np.copy(self.Da)
            eig_old = np.copy(self.eigvecs_a[:Nalpha])

            Vxc_extH_Fock = self.dft_grid_to_fock(vxc_extH, self.Vpot)
            
            Vxc_Fock = Vxc_extH_Fock - vext_opt_Fock - vH0_Fock

            self._diagonalize_with_potential_vFock(v=Vxc_Fock)

            print("Iter: %i, Density Change: %2.2e, Eigenvalue Change: %2.2e."
                  % (OC_step, Derror, eerror))

        # Calculate vxc on grid
        grid_info = self.grid_to_blocks(vxc_grid)
        grid_info[-1].set_pointers(self.wfn.Da())

        vext_opt = self._get_optimized_external_potential(grid_info=grid_info)
        vH0 = self.on_grid_esp(grid=grid_info)[1]
        tauLmP_rho = self._get_l_kinetic_energy_density_directly(self.nt[0], grid_info=grid_info)
        tauP_rho = self._pauli_kinetic_energy_density(self.Da, self.Coca, grid_info=grid_info)
        e_bar = self._average_local_orbital_energy(self.Da, self.Coca, self.eigvecs_a[:Nalpha], grid_info=grid_info)

        self.grid.vxc = e_bar - tauLmP_rho - tauP_rho - vext_opt - vH0
        return self.grid.vxc, e_bar, tauLmP_rho, tauP_rho, vext_opt, vH0