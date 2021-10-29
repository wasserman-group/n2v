"""
mrks.py

Functions associated with mrks inversion
"""

import numpy as np
from opt_einsum import contract
try: 
    import psi4
except:
    pass
import time

class MRKS():
    """
    Wavefunction to KS_potential method based on
    [1] [PRL 115, 083001 (2015)],
    [2] [J. Chem. Phys. 146, 084103 (2017)].

    The XC potential is calculated on the grid
    instead of on the potential basis set.
    Whereas, the guide potential is still used and
    plays the role of v_hartree.
    And because of this, the grid for vxc for output
    has to be specified beforehand.
    
    For CIWavefunction as input, make sure to turn on
    option opdm and tpdm:
        psi4.set_options({
            "opdm": True,
            "tpdm": True,
            'REFERENCE': "RHF"
            })

    Attributes:
    -----------
    Vpot: psi4.core.VBase
        V_potential that contains the info of DFT spherical grid.
    npoint_DFT: int
        number of points for DFT spherical grid.
    vxc_hole_WF: np.ndarray
        vxc_hole_WF on spherical grid. This is stored
        because the calculation of this takes most time.
    """
    vxc_hole_WF = None

    def _vxc_hole_quadrature(self, grid_info=None, atol=1e-5, atol1=1e-4):
        """
        Calculating v_XC^hole in [1] (15) using quadrature
        integral on the default DFT spherical grid.

        Side note: the calculation is actually quite sensitive to atol/atol1
        i.e. the way to handle singularities. Play with it if you
        have a chance.
        The current model is:
        R2 = |r1 - r2|^2
        R2[R2 <= atol] = infinity (so that 1/R = 0)
        R2[atol < R2 < atol1] = atol1

        When self.wfn.name() == CIWavefunction, opdm and tpdm are used
        for calculation;
        when self.wfn.name() == RHF, actually a simplified version (i.e.
        directly using exact HF exchange densities) is implemented. In other
        words, this part can be much quicker and simplifier for RHF by
        obtaining the exchange hole directly as K Matrix without doing
        double integral by quadrature.
        """
        if self.vxc_hole_WF is not None and grid_info is None:
            return self.vxc_hole_WF

        if self.wfn.name() == "CIWavefunction":
            Tau_ijkl = self.wfn.get_tpdm("SUM", True).np
            D2 = self.wfn.get_opdm(-1, -1, "SUM", True).np
            C = self.wfn.Ca()
            Tau_ijkl = contract("pqrs,ip,jq,ur,vs->ijuv", Tau_ijkl, C, C, C, C)
            D2 = C.np @ D2 @ C.np.T
        else:
            D2a = self.wfn.Da().np
            D2b = self.wfn.Db().np
            D2 = D2a + D2b

        if grid_info is None:
            vxchole = np.zeros(self.Vpot.grid().npoints())
            nblocks = self.Vpot.nblocks()
            points_func_outer = self.Vpot.properties()[0]
            blocks = None
        else:
            blocks, npoints, points_func_outer = grid_info
            vxchole = np.zeros(npoints)
            nblocks = len(blocks)
            points_func_outer.set_deriv(0)

        points_func = self.Vpot.properties()[0]
        points_func.set_deriv(0)

        # First loop over the outer set of blocks
        num_block_ten_percent = int(nblocks / 10)
        w1_old = 0
        print(f"vxchole quadrature double integral starts ({(self.Vpot.grid().npoints()):d} points): ", end="")
        start_time = time.time()
        for l_block in range(nblocks):
            # Print out progress
            if num_block_ten_percent != 0 and l_block % num_block_ten_percent == 0:
                print(".", end="")
    
            # Obtain general grid information
            if blocks is None:
                l_grid = self.Vpot.get_block(l_block)
            else:
                l_grid = blocks[l_block]

            l_x = np.array(l_grid.x())
            l_y = np.array(l_grid.y())
            l_z = np.array(l_grid.z())
            l_npoints = l_x.shape[0]

            points_func_outer.compute_points(l_grid)

            l_lpos = np.array(l_grid.functions_local_to_global())
            l_phi = np.array(points_func_outer.basis_values()["PHI"])[:l_npoints, :l_lpos.shape[0]]
    
            # if restricted:
            lD1 = D2[(l_lpos[:, None], l_lpos)]
            rho1 = contract('pm,mn,pn->p', l_phi, lD1, l_phi)
            rho1inv = (1 / rho1)[:, None]

            dvp_l = np.zeros_like(l_x)

            # if not restricted:
            #     dvp_l_b = np.zeros_like(l_x)
    
            # Loop over the inner set of blocks
            for r_block in range(self.Vpot.nblocks()):
                r_grid = self.Vpot.get_block(r_block)
                r_w = np.array(r_grid.w())
                r_x = np.array(r_grid.x())
                r_y = np.array(r_grid.y())
                r_z = np.array(r_grid.z())
                r_npoints = r_w.shape[0]
    
                points_func.compute_points(r_grid)
    
                r_lpos = np.array(r_grid.functions_local_to_global())
    
                # Compute phi!
                r_phi = np.array(points_func.basis_values()["PHI"])[:r_npoints, :r_lpos.shape[0]]
    
                # Build a local slice of D
                if self.wfn.name() == "CIWavefunction":
                    lD2 = D2[(r_lpos[:, None], r_lpos)]
                    rho2 = contract('pm,mn,pn->p', r_phi, lD2, r_phi)
    
                    p, q, r, s = np.meshgrid(l_lpos, l_lpos, r_lpos, r_lpos, indexing="ij")
                    Tap_temp = Tau_ijkl[p, q, r, s]
    
                    n_xc = contract("mnuv,pm,pn,qu,qv->pq", Tap_temp, l_phi, l_phi, r_phi, r_phi)
                    n_xc *= rho1inv
                    n_xc -= rho2
                elif self.wfn.name() == "RHF":
                    lD2 = self.wfn.Da().np[(l_lpos[:, None], r_lpos)]
                    n_xc = - 2 * contract("mu,nv,pm,pn,qu,qv->pq", lD2, lD2, l_phi, l_phi, r_phi, r_phi)
                    n_xc *= rho1inv

                # Build the distnace matrix
                R2 = (l_x[:, None] - r_x) ** 2
                R2 += (l_y[:, None] - r_y) ** 2
                R2 += (l_z[:, None] - r_z) ** 2
                # R2 += 1e-34
                mask1 = R2 <= atol
                mask2 = (R2 > atol) * (R2 < atol1)
                if np.any(mask1 + mask2):
                    R2[mask1] = np.inf
                    R2[mask2] = atol1
                Rinv = 1 / np.sqrt(R2)
    
                # if restricted:
                dvp_l += np.sum(n_xc * Rinv * r_w, axis=1)

            # if restricted:
            vxchole[w1_old:w1_old + l_npoints] += dvp_l
            w1_old += l_npoints
    
        print("\n")
        print(f"Totally {vxchole.shape[0]} grid points takes {(time.time() - start_time):.2f}s "
              f"with max {psi4.core.get_global_option('DFT_BLOCK_MAX_POINTS')} points in a block.")
        assert w1_old == vxchole.shape[0], "Somehow the whole space is not fully integrated."
        if blocks is None:
            # if restricted:
            self.vxc_hole_WF = vxchole
            return self.vxc_hole_WF
        else:
            # if restricted:
            return vxchole

    def _average_local_orbital_energy(self, D, C, eig, grid_info=None):
        """
        (4)(6) in mRKS.
        """

        # Nalpha = self.molecule.nalpha
        # Nbeta = self.molecule.nbeta

        if grid_info is None:
            e_bar = np.zeros(self.Vpot.grid().npoints())
            nblocks = self.Vpot.nblocks()

            points_func = self.Vpot.properties()[0]
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
            lD = D[(l_lpos[:, None], l_lpos)]
            lC = C[l_lpos, :]
            rho = contract('pm,mn,pn->p', l_phi, lD, l_phi)
            e_bar[iw:iw + l_npoints] = contract("pm,mi,ni,i,pn->p", l_phi, lC, lC, eig, l_phi) / rho

            iw += l_npoints
        assert iw == e_bar.shape[0], "Somehow the whole space is not fully integrated."
        return e_bar

    def _pauli_kinetic_energy_density(self, D, C, occ=None, grid_info=None):
        """
        (16)(18) in mRKS. But notice this does not return taup but taup/n
        :return:
        """

        if occ is None:
            occ = np.ones(C.shape[1])

        if grid_info is None:
            taup_rho = np.zeros(self.Vpot.grid().npoints())
            nblocks = self.Vpot.nblocks()

            points_func = self.Vpot.properties()[0]
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

    def mRKS(self, maxiter, vxc_grid=None, v_tol=1e-4, D_tol=1e-7,
             eig_tol=1e-4, frac_old=0.5, init="scan",
             sing=(1e-5, 1e-4, 1e-5, 1e-4)):
        """
        the modified Ryabinkin-Kohut-Staroverov method.

        Currently it supports two different kind of input wavefunction:
            1) Psi4.CIWavefunction
            2) Psi4.RHF
        and it only supports spin-restricted wavefunction.
        Side note: spin-unrestricted HF wavefunction (psi4.UHF) can easily
        be supported but unrestricted CI or restricted/unrestricted CCSD
        can not, because of the absence of tpdm in both methods.

        parameters:
        ----------------------
            maxiter: int
                same as opt_max_iter
            vxc_grid: np.ndarray of shape (3, num_grid_points), opt
                When this is given, the final result will be represented
            v_tol: float, opt
                convergence criteria for vxc Fock matrices.
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
            sing: tuple of float of length 4, opt.
                Singularity parameter for _vxc_hole_quadrature()
                default: (1e-5, 1e-4, 1e-5, 1e-4)
                [0]: atol, [1]: atol1 for dft_spherical grid calculation.
                [2]: atol, [3]: atol1 for vxc_grid calculation.


        returns:
        ----------------------
            The result will be save as self.grid.vxc
    """
        if not self.wfn.name() in ["CIWavefunction", "RHF"]:
            raise ValueError("Currently only supports Psi4 CI wavefunction"
                             "inputs because Psi4 CCSD wavefunction currently "
                             "does not support two-particle density matrices.")

        if self.ref != 1:
            raise ValueError("Currently only supports Spin-Restricted "
                             "calculations since Spin-Unrestricted CI "
                             "is not supported by Psi4.")

        if self.guide_potential_components[0] != "hartree":
            raise ValueError("Hartree potential is necessary as the guide potential.")

        Nalpha = self.nalpha

        # Preparing DFT spherical grid
        functional = psi4.driver.dft.build_superfunctional("SVWN", restricted=True)[0]
        self.Vpot = psi4.core.VBase.build(self.basis, functional, "RV")
        self.Vpot.initialize()
        self.Vpot.properties()[0].set_pointers(self.wfn.Da())


        # Preparing for WF properties
        if self.wfn.name() == "CIWavefunction":
            if not (psi4.core.get_global_option("opdm") and psi4.core.get_global_option("tpdm")):
                raise ValueError("For CIWavefunction as input, make sure to turn on opdm and tpdm.")
            # TPDM & ERI Memory check
            nbf = self.nbf
            I_size = (nbf ** 4) * 8.e-9 * 2
            numpy_memory = 2
            memory_footprint = I_size * 1.5
            if I_size > numpy_memory:
                psi4.core.clean()
                raise Exception("Estimated memory utilization (%4.2f GB) exceeds allotted memory \
                                            limit of %4.2f GB." % (memory_footprint, numpy_memory))
            else:
                print("Memory taken by ERI integral matrix and 2pdm is about: %.3f GB." % memory_footprint)


            opdm = np.array(self.wfn.get_opdm(-1,-1,"SUM",False))
            tpdm = self.wfn.get_tpdm("SUM", True).np

            Ca = self.wfn.Ca().np

            mints = psi4.core.MintsHelper(self.basis)

            Ca_psi4 = psi4.core.Matrix.from_array(Ca)
            I = mints.mo_eri(Ca_psi4, Ca_psi4, Ca_psi4, Ca_psi4).np
            # I = 0.5 * I + 0.25 * np.transpose(I, [0, 1, 3, 2]) + 0.25 * np.transpose(I, [1, 0, 2, 3])
            # Transfer the AO h into MO h
            h = Ca.T @ (self.T + self.V) @ Ca

            # Generalized Fock Matrix is constructed on the
            # basis of MOs, which are orthonormal.
            F_GFM = opdm @ h + contract("rsnq,rsmq->mn", I, tpdm)
            F_GFM = 0.5 * (F_GFM + F_GFM.T)

            del mints, I

            C_a_GFM = psi4.core.Matrix(nbf, nbf)
            eigs_a_GFM = psi4.core.Vector(nbf)
            psi4.core.Matrix.from_array(F_GFM).diagonalize(C_a_GFM,
                                                           eigs_a_GFM,
                                                           psi4.core.DiagonalizeOrder.Ascending)

            eigs_a_GFM = eigs_a_GFM.np / 2.0  # RHF
            C_a_GFM = C_a_GFM.np
            # Transfer to AOs
            C_a_GFM = Ca @ C_a_GFM

            # Solving for Natural Orbitals (NO)>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            C_a_NO = psi4.core.Matrix(nbf, nbf)
            eigs_a_NO = psi4.core.Vector(nbf)
            psi4.core.Matrix.from_array(opdm).diagonalize(C_a_NO, eigs_a_NO, psi4.core.DiagonalizeOrder.Descending)
            eigs_a_NO = eigs_a_NO.np / 2.0  # RHF
            C_a_NO = C_a_NO.np
            C_a_NO = Ca @ C_a_NO

            # prepare properties on the grid
            ebarWF = self._average_local_orbital_energy(self.Dt[0], C_a_GFM, eigs_a_GFM)
            taup_rho_WF = self._pauli_kinetic_energy_density(self.Dt[0], C_a_NO, eigs_a_NO)
        elif self.wfn.name() == "RHF":  # Since HF is a special case, no need for GFM and NO as in CI.
            epsilon_a = self.wfn.epsilon_a_subset("AO", "OCC").np
            ebarWF = self._average_local_orbital_energy(self.Dt[0], self.ct[0][:,:Nalpha], epsilon_a)
            taup_rho_WF = self._pauli_kinetic_energy_density(self.Dt[0], self.ct[0])
        else:
            raise ValueError("Currently only supports Spin-Restricted "
                             "calculations since Spin-Unrestricted CI "
                             "is not supported by Psi4.")

        vxchole = self._vxc_hole_quadrature(atol=sing[0], atol1=sing[1])

        emax = np.max(ebarWF)

        # Initialization.
        if init is None:
            self.Da = np.copy(self.Dt[0])
            self.Coca = np.copy(self.ct[0])
            self.eigvecs_a = self.wfn.epsilon_a().np[:Nalpha]
        elif init.lower()=="continue":
            pass
        else:
            wfn_temp = psi4.energy(init+"/" + self.basis_str, molecule=self.mol, return_wfn=True)[1]
            self.Da = np.array(wfn_temp.Da())
            self.Coca = np.array(wfn_temp.Ca())[:, :Nalpha]
            self.eigvecs_a = np.array(wfn_temp.epsilon_a())
            del wfn_temp

        vxc_old = 0.0
        Da_old = 0.0
        eig_old = 0.0
        for mRKS_step in range(1, maxiter+1):
            # ebarKS = self._average_local_orbital_energy(self.molecule.Da.np, self.molecule.Ca.np[:,:Nalpha], self.molecule.eig_a.np[:Nalpha] + self.vout_constant)
            ebarKS = self._average_local_orbital_energy(self.Da, self.Coca, self.eigvecs_a[:Nalpha])
            taup_rho_KS = self._pauli_kinetic_energy_density(self.Da, self.Coca)
            # self.vout_constant = emax - self.molecule.eig_a.np[self.molecule.nalpha - 1]
            potential_shift = emax - np.max(ebarKS)
            self.shift = potential_shift

            vxc = vxchole + ebarKS - ebarWF + taup_rho_WF - taup_rho_KS + potential_shift

            # Add compulsory mixing parameter close to the convergence to help convergence HOPEFULLY

            verror = np.linalg.norm(vxc - vxc_old) / self.nbf ** 2
            if verror < v_tol:
                print("vxc stops updating.")
                break

            Derror = np.linalg.norm(self.Da - Da_old) / self.nbf ** 2
            eerror = (np.linalg.norm(self.eigvecs_a[:Nalpha] - eig_old) / Nalpha)
            if (Derror < D_tol) and (eerror < eig_tol):
                print("KSDFT stops updating.")
                break

            # linear Mixture
            if mRKS_step != 1:
                vxc = vxc * (1 - frac_old) + vxc_old * frac_old

            # Save old data.
            vxc_old = np.copy(vxc)
            Da_old = np.copy(self.Da)
            eig_old = np.copy(self.eigvecs_a[:Nalpha])

            vxc_Fock = self.dft_grid_to_fock(vxc, self.Vpot)

            self._diagonalize_with_potential_vFock(v=vxc_Fock)

            print(f"Iter: {mRKS_step}, Density Change: {Derror:2.2e}, Eigenvalue Change: {eerror:2.2e}, "
                  f"Potential Change: {verror:2.2e}.")

        if vxc_grid is not None:
            grid_info = self.grid_to_blocks(vxc_grid)
            grid_info[-1].set_pointers(self.wfn.Da())

            # A larger atol seems to be necessary for user-defined grid
            vxchole = self._vxc_hole_quadrature(grid_info=grid_info,
                                                atol=sing[2], atol1=sing[3])
            if self.wfn.name() == "CIWavefunction":
                ebarWF = self._average_local_orbital_energy(self.Dt[0],
                                                            C_a_GFM, eigs_a_GFM,
                                                            grid_info=grid_info)
                taup_rho_WF = self._pauli_kinetic_energy_density(self.Dt[0],
                                                                 C_a_NO, eigs_a_NO,
                                                                 grid_info=grid_info)
            elif self.wfn.name() == "RHF":
                ebarWF = self._average_local_orbital_energy(self.Dt[0],
                                                            self.ct[0],
                                                            epsilon_a[:Nalpha],
                                                            grid_info=grid_info)
                taup_rho_WF = self._pauli_kinetic_energy_density(self.Dt[0],
                                                                 self.ct[0],
                                                                 grid_info=grid_info)
            ebarKS = self._average_local_orbital_energy(self.Da, self.Coca,
                                                        self.eigvecs_a[:Nalpha], grid_info=grid_info)
            taup_rho_KS = self._pauli_kinetic_energy_density(self.Da, self.Coca,
                                                             grid_info=grid_info)

            potential_shift = np.max(ebarWF) - np.max(ebarKS)
            self.shift = potential_shift

            vxc = vxchole + ebarKS - ebarWF + taup_rho_WF - taup_rho_KS + potential_shift
        self.grid.vxc = vxc
        return

    def _diagonalize_with_potential_vFock(self, v=None):
        """
        Diagonalize Fock matrix with additional external potential
        """

        if v is None:
            fock_a = self.V + self.T + self.va
        else:
            if self.ref == 1:
                fock_a = self.V + self.T + self.va + v
            else:
                valpha, vbeta = v
                fock_a = self.V + self.T + self.va + valpha
                fock_b = self.V + self.T + self.vb + vbeta


        self.Ca, self.Coca, self.Da, self.eigvecs_a = self.diagonalize( fock_a, self.nalpha )

        if self.ref == 1:
            self.Cb, self.Cocb, self.Db, self.eigvecs_b = self.Ca.copy(), self.Coca.copy(), self.Da.copy(), self.eigvecs_a.copy()
        else:
            self.Cb, self.Cocb, self.Db, self.eigvecs_b = self.diagonalize( fock_b, self.nbeta )