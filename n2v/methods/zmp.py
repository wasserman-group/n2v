"""
zmp.py

Functions associated with zmp inversion
"""

import psi4
import numpy as np
from opt_einsum import contract
from scipy.optimize import minimize
from functools import reduce
import sys

import matplotlib.pyplot as plt

psi4.core.be_quiet()
eps = np.finfo(float).eps

class ZMP():
    def zmp(self, 
            lambda_list,
            zmp_kernel,
            opt_max_iter, 
            opt_tol, 
            ):

        """
        Performs ZMP optimization according to: 
        
        1) 'From electron densities to Kohn-Sham kinetic energies, orbital energies,
        exchange-correlation potentials, and exchange-correlation energies' by
        Zhao + Morrison + Parr. 
        https://doi.org/10.1103/PhysRevA.50.2138

        Additional DIIS algorithms obtained from:
        2) 'Psi4NumPy: An interactive quantum chemistry programming environment 
        for reference implementations and rapid development.' by 
        Daniel G.A. Smith and others. 
        https://doi.org/10.1021/acs.jctc.8b00286

        Functionals that drive the SCF procedure are obtained from:
        https://doi.org/10.1002/qua.26400

        Parameters:
        -----------
        lam: int
            Lamda parameter used as a coefficient for Hartree 
            difference in SCF cycle. 
        zmp_kernel: str
            Specifies what functional to use to drive the SCF procedure.
            Options: {'hartree', 'log', 'exp', 'grad'}
        opt_max_iter: float
            Maximum number of iterations for scf cycle
        opt_tol: float
            Convergence criteria set for Density Difference and DIIS error. 
        """


        self.diis_space = 200

        print("Running SCF ZMP:")
        self.zmp_scf(lambda_list, zmp_kernel, opt_max_iter, D_conv=opt_tol)

    def zmp_scf(self, 
            lambda_list = [1],
            zmp_kernel = 'hartree',
            maxiter = 100, 
            D_conv = psi4.core.get_option("SCF", "D_CONVERGENCE")):
        """
        Performs scf cycle
        Parameters
        ----------
        lam: integer, opt
            Global lagrange multiplier for effective potential that drives SCF calculation. 
            Defined in equation 7 and 8 of reference (1).
        maxiter: integer, opt
            Maximum number of iterations for SCF calculation
        D_conv: float
            Convergence parameter for density and DIIS error. 
            Default is Psi4's Density convergence parameter: 1e-06
        """

        A = np.array(self.A)
        S2 = self.S2
        H = self.T + self.V 

        # Trial & Residual Vector Lists
        state_vectors_a, state_vectors_b = [], []
        error_vectors_a, error_vectors_b = [], []

        if hasattr(self.wfn, 'V_potential()') == False:
            _, dft_wfn = psi4.energy('svwn'+'/'+self.basis_str, molecule=self.mol, return_wfn=True)
        else:
            dft_wfn = self.wfn

        #"Initial Guess for SCF
        Cocca = self.ct[0]
        Coccb = self.ct[1]
        Da = self.nt[0]
        Db = self.nt[1]
        self.Da = self.nt[0]
        self.Db = self.nt[1]
        D_old = Da.copy()

        # Obtain slice of target density to test convergence on grid
        # at the end of scf. 
        x = np.linspace(-0,15,201)
        y = np.linspace(-0,15,201)
        z = np.linspace(-0,15,201)
        grid = np.concatenate((x[:,None], y[:,None], z[:,None]), axis=1).T
        density0 = self.on_grid_density(grid, Da=self.nt[0], Db=self.nt[1])

        dd_old = 0.0
        lam_iter = True
        vc_global = np.zeros((self.nbf, self.nbf))
        vxc_sum = None
        list_of_vxc = []

        self.protosum_a = np.zeros_like(Da)
        self.protosum_b = np.zeros_like(Db)

        for lam_i in lambda_list:
            self.shift = 0.1 * lam_i
            
            Cocca = self.ct[0]
            Coccb = self.ct[1]
            state_vectors_a, state_vectors_b = [], []
            error_vectors_a, error_vectors_b = [], []

            print(f"-------------SCF TIME ----------------------Current Lamda: {lam_i}")
            for SCF_ITER in range(1,maxiter):

    #------------->  Generate Fock Matrix:
                Fa = np.zeros((self.nbf, self.nbf))
                Fb = np.zeros((self.nbf, self.nbf))
                J, _ = self.form_jk(Cocca, Coccb)

                if zmp_kernel.lower() == 'hartree':
                    #Equation 7 of Reference (1)
                    v_c = lam_i * ( (J[0] + J[1]) - (self.J0[0] + self.J0[1]) )
                else:
                    #Equations 37, 38, 39, 40 of Reference (3)
                    v_c = self.generate_s_fucntional(lam, zmp_kernel)

                #Equation 10 of Reference (1)
                Fa += H + self.va + v_c + vc_global
                Fb += H + self.vb + v_c + vc_global

                #Level Shift
                Fa += (S2 - reduce(np.dot, (S2, Da, S2)) * self.shift)
                Fb += (S2 - reduce(np.dot, (S2, Db, S2)) * self.shift)

    #------------->  DIIS:
                if SCF_ITER > 1:
                    #Construct the AO gradient
                    # r = (A(FDS - SDF)A)_mu_nu
                    grad_a = A.dot(Fa.dot(Da).dot(S2) - S2.dot(Da).dot(Fa)).dot(A)
                    grad_a[np.abs(grad_a) < eps] = 0.0

                    if SCF_ITER -1 < self.diis_space:
                        state_vectors_a.append(Fa.copy())
                        error_vectors_a.append(grad_a.copy())
                    else:
                        state_vectors_a.append(Fa.copy())
                        error_vectors_a.append(grad_a.copy())

                    #Build inner product of error vectors
                    Bdim = len(state_vectors_a)
                    Ba = np.empty((Bdim + 1, Bdim + 1))
                    Ba[-1, :] = -1
                    Ba[:, -1] = -1
                    Ba[-1, -1] = 0
                    Bb = Ba.copy()

                    for i in range(len(state_vectors_a)):
                        for j in range(len(state_vectors_a)):
                            Ba[i,j] = np.einsum('ij,ij->', error_vectors_a[i], error_vectors_a[j])

                    #Build almost zeros matrix to generate inverse. 
                    RHS = np.zeros(Bdim + 1)
                    RHS[-1] = -1

                    #Find coefficient matrix:
                    Ca = np.linalg.solve(Ba, RHS.copy())
                    Ca[np.abs(Ca) < eps] = 0.0 

                    #Generate new fock Matrix:
                    Fa = np.zeros_like(Fa)
                    for i in range(Ca.shape[0] - 1):
                        Fa += Ca[i] * state_vectors_a[i]

                    diis_error_a = np.max(np.abs(error_vectors_a[-1]))
                    if self.ref ==  1:
                        Fb = Fa.copy()
                        diis_error = 2 * diis_error_a

                    else:
                        grad_b = A.dot(Fb.dot(Db).dot(S2) - S2.dot(Db).dot(Fb)).dot(A)
                        grad_b[np.abs(grad_b) < eps] = 0.0
                        
                        if SCF_ITER -1 < self.diis_space:
                            state_vectors_b.append(Fb.copy())
                            error_vectors_b.append(grad_b.copy())
                        else:
                            state_vectors_b.append(Fb.copy())
                            error_vectors_b.append(grad_b.copy())

                        for i in range(len(state_vectors_b)):
                            for j in range(len(state_vectors_b)):
                                Bb[i,j] = np.einsum('ij,ij->', error_vectors_b[i], error_vectors_b[j])

                        diis_error_b = np.max(np.abs(error_vectors_b[-1]))
                        diis_error = diis_error_a + diis_error_b

                        Cb = np.linalg.solve(Bb, RHS.copy())
                        Cb[np.abs(Cb) < eps] = 0.0 

                        Fb = np.zeros_like(Fb)
                        for i in range(Cb.shape[0] - 1):
                            Fb += Cb[i] * state_vectors_b[i]

                else:
                    diis_error = 1.0
                
    #------------->  Diagonalization | Check convergence:

                Ca, Cocca, Da, eigs_a = self.diagonalize(Fa, self.nalpha)
                if self.ref == 2:
                    Cb, Coccb, Db, eigs_b = self.diagonalize(Fb, self.nbeta)
                else: 
                    Cb, Coccb, Db, eigs_b = Ca.copy(), Cocca.copy(), Da.copy(), eigs_a.copy()

                ddm = D_old - Da
                D_old = Da
                derror = np.max(np.abs(ddm))

                if True and np.mod(SCF_ITER,5) == 0.0:
                    print(f"Iteration: {SCF_ITER:d} | Self Convergence Error: {derror:10.5e} | DIIS Error: {diis_error:10.5e}")
                if abs(derror) < D_conv and abs(diis_error) < D_conv:
                    break
                if SCF_ITER == maxiter - 1:
                    raise ValueError("Maximum Number of SCF cycles reached. Try different settings.")


            density_current = self.on_grid_density(grid, Da=Da, Db=Db)
            grid_diff = np.max(np.abs(density0 - density_current))
            print(f"SCF Procedure successfull. Max density difference: {grid_diff}")
            self.Da = Da
            self.Db = Db

            #VXC is hartree-like Potential. We remove Fermi_Amaldi Guess. 
            self.proto_density_a = lam_i * (self.Da) - (lam_i + 1/(self.nalpha + self.nbeta)) * (self.nt[0])
            self.proto_density_b = lam_i * (self.Db) - (lam_i + 1/(self.nbeta + self.nalpha)) * (self.nt[1])

            self.protosum_a += self.proto_density_a.copy()
            self.protosum_b += self.proto_density_b.copy()

            self.wfn.Da().np[:] = self.protosum_a
            self.wfn.Db().np[:] = self.protosum_b

            potentials = self.on_grid_esp(vpot=dft_wfn.V_potential())
            vxc_current = potentials[1]
            vxc_sum = vxc_sum + vxc_current if vxc_sum is not None else vxc_current
            vc_i = self.dft_grid_to_fock(vxc_sum, Vpot=dft_wfn.V_potential())
            vc_global += vc_i

            # #Printing GRID
            # npoints=401
            # x = np.linspace(-0,15,npoints)[:,None]
            # y = np.linspace(-0,15,npoints)[:,None]
            # z = np.linspace(-0,15,npoints)[:,None]
            # grid_0 = np.concatenate((x,y,z), axis=1).T
            # results = self.on_grid_esp(grid=grid_0, )
            # vxc_be = results[1]

            # plt.plot(x,  vxc_be, label="vxc")
            # plt.xscale('log')
            # plt.legend()
            # plt.show()




    def generate_s_fucntional(self, lam, zmp_kernel):
        """
        Generates S_n Functional as described in:
        https://doi.org/10.1002/qua.26400
        """

        #Get Density on the grid
        if self.wfn.V_potential() == None:
            _, dft_wfn = psi4.energy('svwn'+'/'+self.basis_str, molecule=self.mol, return_wfn=True)
        else:
            dft_wfn = self.wfn

        D = self.on_grid_density(vpot=dft_wfn.V_potential())
        D0 = self.on_grid_density(grid=None, Da=self.nt[0], Db=self.nt[1],  vpot=dft_wfn.V_potential())

        if self.ref == 1:
            Da, Db = D[:,0], D[:,0]
            Da0, Db0 = D0[:,0], D0[:,0]
        else:
            Da, Db = D[:, 0], D[:, 1]
            Da0, Db0 = D0[:,0], D0[:,1]
            
        #Calculate kernels
        if zmp_kernel == 'log':
            Sa = np.log(Da) + 1
            Sb = np.log(Db) + 1 
            Sa0 = np.log(Da0) + 1  
            Sb0 = np.log(Db0) + 1
        if zmp_kernel == 'exp':
            Sa = Da ** 0.05
            Sb = Db ** 0.05
            Sa0 = Da0 ** 0.05
            Sb0 = Db0 ** 0.05

        #Generate AO matrix from functions. 
        Sa = self.dft_grid_to_fock(Sa, Vpot=dft_wfn.V_potential())
        Sb = self.dft_grid_to_fock(Sb, Vpot=dft_wfn.V_potential())
        Sa0 = self.dft_grid_to_fock(Sa0, Vpot=dft_wfn.V_potential())
        Sb0 = self.dft_grid_to_fock(Sb0, Vpot=dft_wfn.V_potential())
        v_c = (Sa + Sb) - (Sa0 + Sb0)

        return v_c
        