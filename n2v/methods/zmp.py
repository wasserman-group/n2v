"""
zmp.py

Functions associated with zmp inversion
"""

import psi4
import numpy as np
from functools import reduce

psi4.core.be_quiet()
eps = np.finfo(float).eps

class ZMP():
    def zmp(self, 
            opt_max_iter=100, 
            opt_tol= psi4.core.get_option("SCF", "D_CONVERGENCE"), 
            lambda_list=[70],
            zmp_functional='hartree',
            zmp_mixing = 1,
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
        lambda_list: list
            List of Lamda parameters used as a coefficient for Hartree 
            difference in SCF cycle. 
        zmp_functional: str
            Specifies what functional to use to drive the SCF procedure.
            Options: {'hartree', 'log', 'exp', 'grad'}
        mixing: float
            mixing \in [0,1]. How much of the new potential is added in 
            a given scf step. Values close to 1 may prevent SCF to convergence. 
        opt_max_iter: float
            Maximum number of iterations for scf cycle
        opt_tol: float
            Convergence criteria set for Density Difference and DIIS error. 
        """
        self.diis_space = 100
        self.mixing = zmp_mixing

        print("\nRunning ZMP:")
        self.zmp_scf(lambda_list, zmp_functional, opt_max_iter, D_conv=opt_tol)

    def zmp_scf(self, 
            lambda_list,
            zmp_functional,
            maxiter, 
            D_conv):
        """
        Performs scf cycle
        Parameters
        ----------
        lam_list: list, opt
            Global set lagrange multiplier for effective potential that drives SCF calculation. 
            Defined in equation 7 and 8 of reference (1).
        zmp_functional: str
            Specifies what functional to use to drive the SCF procedure.
            Options: {'hartree', 'log', 'exp', 'grad'}
        maxiter: integer, opt
            Maximum number of iterations for SCF calculation
        D_conv: float
            Convergence parameter for density and DIIS error. 
            Default is Psi4's Density convergence parameter: 1e-06
        """

        #Checks if there is dft grid available. 
        if hasattr(self.wfn, 'V_potential()') == False:
            ref = "RV" if self.ref == 1 else "UV"
            functional = psi4.driver.dft.build_superfunctional("SVWN", restricted=True if self.ref==1 else False)[0]
            Vpot = psi4.core.VBase.build(self.basis, functional, ref)
            Vpot.initialize()
            self.vpot = Vpot
        else:
            self.vpot = self.wfn.V_potential()

        # Obtain target density on dft_grid 
        D0 = self.on_grid_density(grid=None, Da=self.nt[0], Db=self.nt[1],  vpot=self.vpot)

        #Calculate kernels
        if zmp_functional.lower() != 'hartree':
            if self.ref == 1:
                Da0, Db0 = D0[:,0], D0[:,0]
            else:
                Da0, Db0 = D0[:,0], D0[:,1]

            if zmp_functional == 'log':
                Sa0 = np.log(Da0) + 1  
                Sb0 = np.log(Db0) + 1
            elif zmp_functional == 'exp':
                Sa0 = Da0 ** 0.05
                Sb0 = Db0 ** 0.05
            self.Sa0 = self.dft_grid_to_fock(Sa0, Vpot=self.vpot)
            if self.ref == 2:
                self.Sb0 = self.dft_grid_to_fock(Sb0, Vpot=self.vpot)
            else: 
                self.Sb0 = self.Sa0.copy()

        vc_previous = np.zeros((self.nbf, self.nbf))
        self.Da = self.nt[0]
        self.Db = self.nt[1]
        Da = self.nt[0]
        Db = self.nt[1]
        Cocca = self.ct[0]
        Coccb = self.ct[1]

        grid_diff_old = 1/np.finfo(float).eps
        self.proto_density_a =  np.zeros_like(Da)
        self.proto_density_b =  np.zeros_like(Db)

#------------->  Iterating over lambdas:
        for lam_i in lambda_list:
            self.shift = 0.1 * lam_i
            D_old = self.nt[0]

            # Trial & Residual Vector Lists
            state_vectors_a, state_vectors_b = [], []
            error_vectors_a, error_vectors_b = [], []

            for SCF_ITER in range(1,maxiter):

#------------->  Generate Fock Matrix:
                vc = self.generate_s_functional(lam_i,
                                                zmp_functional,
                                                Cocca, Coccb, 
                                                Da, Db)

                # #Potential mixing
                # vc = (1-self.mixing) * vc + (self.mixing) * vc_old

                #Equation 10 of Reference (1). Level shift. 
                Fa = self.T + self.V + self.va + vc + vc_previous
                Fa += (self.S2 - reduce(np.dot, (self.S2, Da, self.S2))) * self.shift

                if self.ref == 2:
                    Fb = self.T + self.V + self.vb + vc + vc_previous
                    Fb += (self.S2 - reduce(np.dot, (self.S2, Db, self.S2))) * self.shift
                

    #------------->  DIIS:
                if SCF_ITER > 1:
                    #Construct the AO gradient
                    # r = (A(FDS - SDF)A)_mu_nu
                    grad_a = self.A.dot(Fa.dot(Da).dot(self.S2) - self.S2.dot(Da).dot(Fa)).dot(self.A)
                    grad_a[np.abs(grad_a) < eps] = 0.0

                    if SCF_ITER < self.diis_space:
                        state_vectors_a.append(Fa.copy())
                        error_vectors_a.append(grad_a.copy())
                    else:
                        state_vectors_a.append(Fa.copy())
                        error_vectors_a.append(grad_a.copy())
                        del state_vectors_a[-1]
                        del error_vectors_a[-1]

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
                        grad_b = self.A.dot(Fb.dot(Db).dot(self.S2) - self.S2.dot(Db).dot(Fb)).dot(self.A)
                        grad_b[np.abs(grad_b) < eps] = 0.0
                        
                        if SCF_ITER < self.diis_space:
                            state_vectors_b.append(Fb.copy())
                            error_vectors_b.append(grad_b.copy())
                        else:
                            state_vectors_b.append(Fb.copy())
                            error_vectors_b.append(grad_b.copy())
                            del state_vectors_b[-1]
                            del error_vectors_b[-1]

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

                #Uncomment to debug internal SCF
                # if len(lambda_list==1) and np.mod(SCF_ITER,5) == 0.0:
                #    print(f"Iteration: {SCF_ITER:3d} | Self Convergence Error: {derror:10.5e} | DIIS Error: {diis_error:10.5e}")
                
                if abs(derror) < D_conv and abs(diis_error) < D_conv:
                    break
                if SCF_ITER == maxiter - 1:
                    raise ValueError("Maximum Number of SCF cycles reached. Try different settings.")

            density_current = self.on_grid_density(grid=None, Da=Da, Db=Db, vpot=self.vpot)
            grid_diff = np.max(np.abs(D0 - density_current))
            if np.abs(grid_diff_old) < np.abs(grid_diff):
                print(f"\nZMP halted. Density Error Stops Updating: old: {grid_diff_old}, current: {grid_diff}.")
                break

            grid_diff_old = grid_diff
            print(f"SCF Converged for lambda:{int(lam_i):5d}. Max density difference: {grid_diff}")
            #VXC is hartree-like Potential. We remove Fermi_Amaldi Guess.
            self.proto_density_a += lam_i * (Da - self.nt[0]) * self.mixing
            if self.ref == 2:
                self.proto_density_b += lam_i * (Db - self.nt[1]) * self.mixing
            else:
                self.proto_density_b = self.proto_density_a.copy()
            vc_previous += vc * self.mixing
            # REAL LINEAR MIXING
            # if not np.isclose(np.linalg.norm(vc_previous), 0): # dont mix for first lambda
            #     self.proto_density_a = lam_i * (Da - self.nt[0]) * (1-self.mixing) + self.proto_density_a * self.mixing
            #     self.proto_density_b = lam_i * (Db - self.nt[1]) * (1-self.mixing) + self.proto_density_b * self.mixing
            #     vc_previous = vc * (1-self.mixing) + vc_previous * self.mixing
            # else:
            #     vc_previous = vc
            #     self.proto_density_a = lam_i * (Da - self.nt[0])
            #     self.proto_density_b = lam_i * (Db - self.nt[1])

        self.proto_density_a -= lam_i * (Da - self.nt[0]) * self.mixing
        self.proto_density_a += lam_i * (Da - self.nt[0])
        self.proto_density_a -= (1 / (self.nalpha + self.nbeta)) * (self.nt[0])
        self.Da = Da
        self.Ca = Ca
        self.Coca = Cocca

        if self.ref == 2:
            self.proto_density_b -= lam_i * (Db - self.nt[1]) * self.mixing
            self.proto_density_b += lam_i * (Db - self.nt[1])
            self.proto_density_b -= (1 / (self.nbeta + self.nalpha)) * (self.nt[1])
            self.Db = Db
            self.Cb = Cb
            self.Cocb = Coccb

        else:
            self.proto_density_b = self.proto_density_a.copy()
            self.Db = self.Da.copy()
            self.Cb = self.Ca.copy()
            self.Cocb = self.Coca.copy()


    def generate_s_functional(self, lam, zmp_functional, Cocca, Coccb, Da, Db):
        """
        Generates S_n Functional as described in:
        https://doi.org/10.1002/qua.26400
        """

        if zmp_functional.lower() == 'hartree':
            J, _ = self.form_jk(Cocca, Coccb)

            #Equation 7 of Reference (1)
            vc = lam * ( (J[0] + J[1]) - (self.J0[0] + self.J0[1]) )
            return vc

        else:

            D = self.on_grid_density(vpot=self.vpot, Da=Da, Db=Db)

            if self.ref == 1:
                Da, Db = D[:,0], D[:,0]
            else:
                Da, Db = D[:,0], D[:,1]
                
            #Calculate kernels
            #Equations 37, 38, 39, 40 of Reference (3)
            if zmp_functional.lower() == 'log':
                Sa = np.log(Da) + 1
                Sb = np.log(Db) + 1 
            if zmp_functional.lower() == 'exp':
                Sa = Da ** 0.05
                Sb = Db ** 0.05

            #Generate AO matrix from functions. 
            Sa = self.dft_grid_to_fock(Sa, Vpot=self.vpot)
            Sb = self.dft_grid_to_fock(Sb, Vpot=self.vpot)
            vc = (Sa + Sb) - (self.Sa0 + self.Sb0)

        return vc
        