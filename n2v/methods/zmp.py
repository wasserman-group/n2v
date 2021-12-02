"""
zmp.py

Functions associated with zmp inversion
"""


import psi4
psi4.core.be_quiet()
import numpy as np
from functools import reduce


eps = np.finfo(float).eps

class ZMP():
    def zmp(self, 
            opt_max_iter=100, 
            opt_tol= psi4.core.get_option("SCF", "D_CONVERGENCE"), 
            lambda_list=[70],
            zmp_mixing = 1,
            print_scf = False, 
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
        zmp_mixing: float, optional
            mixing \in [0,1]. How much of the new potential is added in.
            For example, zmp_mixing = 0 means the traditional ZMP, i.e. all the potentials from previous
            smaller lambda are ignored.
            Zmp_mixing = 1 means that all the potentials of previous lambdas are accumulated, the larger lambda
            potential are meant to fix the wrong/inaccurate region of the potential of the sum of the previous
            potentials instead of providing an entire new potentials.
            default: 1
        opt_max_iter: float
            Maximum number of iterations for scf cycle
        opt_tol: float
            Convergence criteria set for Density Difference and DIIS error.
        return:
            The result will be stored in self.proto_density_a and self.proto_density_b
            For zmp_mixing==1, restricted (ref==1):
                self.proto_density_a = \sum_i lambda_i * (Da_i - Dt[0]) - 1/N * (Dt[0])
                self.proto_density_b = \sum_i lambda_i * (Db_i - Dt[1]) - 1/N * (Dt[1]);
            unrestricted (ref==1):
                self.proto_density_a = \sum_i lambda_i * (Da_i - Dt[0]) - 1/N * (Dt[0] + Dt[1])
                self.proto_density_b = \sum_i lambda_i * (Db_i - Dt[1]) - 1/N * (Dt[0] + Dt[1]);
            For restricted (ref==1):
                vxc = \int dr' \frac{self.proto_density_a + self.proto_density_b}{|r-r'|}
                    = 2 * \int dr' \frac{self.proto_density_a}{|r-r'|};
            for unrestricted (ref==2):
                vxc_up = \int dr' \frac{self.proto_density_a}{|r-r'|}
                vxc_down = \int dr' \frac{self.proto_density_b}{|r-r'|}.
            To get potential on grid, one needs to do
                vxc = self.on_grid_esp(Da=self.proto_density_a, Db=self.proto_density_b, grid=grid) for restricted;
                vxc_up = self.on_grid_esp(Da=self.proto_density_a, Db=np.zeros_like(self.proto_density_a),
                          grid=grid) for unrestricted;
        """
        self.diis_space = 100
        self.mixing = zmp_mixing

        print("\nRunning ZMP:")
        self.zmp_scf(lambda_list, 'hartree', opt_max_iter, print_scf, D_conv=opt_tol)

    def zmp_scf(self, 
            lambda_list,
            zmp_functional,
            maxiter, 
            print_scf,
            D_conv):
        """
        Performs scf cycle
        Parameters:
            zmp_functional: options the penalty term.
            But others are not currently working except for Hartree penalty (original ZMP).
        ----------
        """

        #Checks if there is dft grid available. 
        if hasattr(self.wfn, 'V_potential()') == False:
            ref = "RV" if self.ref == 1 else "UV"
            functional = psi4.driver.dft.build_superfunctional("SVWN", restricted=True if self.ref==1 else False)[0]
            Vpot = psi4.core.VBase.build(self.basis, functional, ref)
            Vpot.initialize()
            self.Vpot = Vpot
        else:
            self.Vpot = self.wfn.V_potential()

        # Obtain target density on dft_grid 
        D0 = self.on_grid_density(grid=None, Da=self.Dt[0], Db=self.Dt[1],  Vpot=self.Vpot)

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
            self.Sa0 = self.dft_grid_to_fock(Sa0, Vpot=self.Vpot)
            if self.ref == 2:
                self.Sb0 = self.dft_grid_to_fock(Sb0, Vpot=self.Vpot)
            else: 
                self.Sb0 = self.Sa0.copy()

        vc_previous_a = np.zeros((self.nbf, self.nbf))
        vc_previous_b = np.zeros((self.nbf, self.nbf))
        self.Da = self.Dt[0]
        self.Db = self.Dt[1]
        Da = self.Dt[0]
        Db = self.Dt[1]
        Cocca = self.ct[0]
        Coccb = self.ct[1]

        grid_diff_old = 1/np.finfo(float).eps
        self.proto_density_a =  np.zeros_like(Da)
        self.proto_density_b =  np.zeros_like(Db)

#------------->  Iterating over lambdas:
        for lam_i in lambda_list:
            self.shift = 0.1 * lam_i
            D_old = self.Dt[0]

            # Trial & Residual Vector Lists
            state_vectors_a, state_vectors_b = [], []
            error_vectors_a, error_vectors_b = [], []

            for SCF_ITER in range(1,maxiter):

#------------->  Generate Fock Matrix:
                vc = self.generate_s_functional(lam_i,
                                                zmp_functional,
                                                Cocca, Coccb, 
                                                Da, Db)

                #Equation 10 of Reference (1). Level shift. 
                Fa = self.T + self.V + self.va + vc[0] + vc_previous_a
                Fa += (self.S2 - reduce(np.dot, (self.S2, Da, self.S2))) * self.shift

                if self.ref == 2:
                    Fb = self.T + self.V + self.vb + vc[1] + vc_previous_b
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
                        del state_vectors_a[0]
                        del error_vectors_a[0]

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
                            del state_vectors_b[0]
                            del error_vectors_b[0]

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
                if print_scf is True:
                    if np.mod(SCF_ITER,5) == 0.0:
                        print(f"Iteration: {SCF_ITER:3d} | Self Convergence Error: {derror:10.5e} | DIIS Error: {diis_error:10.5e}")
                
                #DIIS error may improve as fast as the D_conv. Relax the criteria an order of magnitude. 
                if abs(derror) < D_conv and abs(diis_error) < D_conv*10: 
                    break
                if SCF_ITER == maxiter - 1:
                    raise ValueError("ZMP Error: Maximum Number of SCF cycles reached. Try different settings.")

            density_current = self.on_grid_density(grid=None, Da=Da, Db=Db, Vpot=self.Vpot)
            grid_diff = np.max(np.abs(D0 - density_current))
            if np.abs(grid_diff_old) < np.abs(grid_diff):
                # This is a greedy algorithm: if the density error stopped improving for this lambda, we will stop here.
                print(f"\nZMP halted at lambda={lam_i}. Density Error Stops Updating: old: {grid_diff_old}, current: {grid_diff}.")
                break

            grid_diff_old = grid_diff
            print(f"SCF Converged for lambda:{int(lam_i):5d}. Max density difference: {grid_diff}")

            self.proto_density_a += lam_i * (Da - self.Dt[0]) * self.mixing
            if self.ref == 2:
                self.proto_density_b += lam_i * (Db - self.Dt[1]) * self.mixing
            else:
                self.proto_density_b = self.proto_density_a.copy()

            vc_previous_a += vc[0] * self.mixing
            if self.ref == 2:
                vc_previous_b += vc[1] * self.mixing

            # this is the lambda that is already proven to be improving the density, i.e. the corresponding
            # potential has updated to proto_density
            successful_lam = lam_i
            # The proto_density corresponds to successful_lam
            successful_proto_density = [(Da - self.Dt[0]), (Db - self.Dt[1])]
# -------------> END Iterating over lambdas:

        self.proto_density_a += successful_lam * successful_proto_density[0] * (1 - self.mixing)
        if self.guide_potential_components[0].lower() == "fermi_amaldi":
            # for ref==1, vxc = \int dr (proto_density_a + proto_density_b)/|r-r'| - 1/N*vH
            if self.ref == 1:
                self.proto_density_a -= (1 / (self.nalpha + self.nbeta)) * (self.Dt[0])
            # for ref==1, vxc = \int dr (proto_density_a)/|r-r'| - 1/N*vH
            else:
                self.proto_density_a -= (1 / (self.nalpha + self.nbeta)) * (self.Dt[0] + self.Dt[1])

        self.Da = Da
        self.Ca = Ca
        self.Coca = Cocca
        self.eigvecs_a = eigs_a

        if self.ref == 2:
            self.proto_density_b += successful_lam * successful_proto_density[1] * (1 - self.mixing)
            if self.guide_potential_components[0].lower() == "fermi_amaldi":
                # for ref==1, vxc = \int dr (proto_density_a + proto_density_b)/|r-r'| - 1/N*vH
                if self.ref == 1:
                    self.proto_density_b -= (1 / (self.nalpha + self.nbeta)) * (self.Dt[1])
                # for ref==1, vxc = \int dr (proto_density_a)/|r-r'| - 1/N*vH
                else:
                    self.proto_density_b -= (1 / (self.nalpha + self.nbeta)) * (self.Dt[0] + self.Dt[1])
            self.Db = Db
            self.Cb = Cb
            self.Cocb = Coccb
            self.eigvecs_b = eigs_b

        else:
            self.proto_density_b = self.proto_density_a.copy()
            self.Db = self.Da.copy()
            self.Cb = self.Ca.copy()
            self.Cocb = self.Coca.copy()
            self.eigvecs_b = self.eigvecs_a.copy()
        print(successful_lam)


    def generate_s_functional(self, lam, zmp_functional, Cocca, Coccb, Da, Db):
        """
        Generates S_n Functional as described in:
        https://doi.org/10.1002/qua.26400
        """

        if zmp_functional.lower() == 'hartree':
            J, _ = self.form_jk(Cocca, Coccb)

            #Equation 7 of Reference (1)
            if self.ref == 1:
                vc_a = 2 * lam * ( J[0] - self.J0[0] ) 
                vc = [vc_a]
            else:
                vc_a = lam * ( J[0] - self.J0[0] ) 
                vc_b = lam * ( J[1] - self.J0[1] )
                vc = [vc_a, vc_b]            
            return vc

        else:

            D = self.on_grid_density(Vpot=self.Vpot, Da=Da, Db=Db)

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
            Sa = self.dft_grid_to_fock(Sa, Vpot=self.Vpot)
            Sb = self.dft_grid_to_fock(Sb, Vpot=self.Vpot)
            vc = (Sa + Sb) - (self.Sa0 + self.Sb0)

        return vc
        
