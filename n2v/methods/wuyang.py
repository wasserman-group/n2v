"""
wuyang.py

Functions associated with wuyang inversion
"""

import numpy as np
from opt_einsum import contract
from scipy.optimize import minimize

class WuYang():
    """
    Performs Optimization as in: 10.1063/1.1535422 - Qin Wu + Weitao Yang
    """

    def wuyang(self, opt_method):
        """
        Calls scipy minimizer to minimize lagrangian. 
        """

        if opt_method == 'bfgs':
            opt_results = minimize( fun = self.lagrangian,
                                    x0  = self.v0, 
                                    jac = self.gradient,
                                    method = opt_method,
                                    tol    = 1e-7,
                                    options = {"maxiter" : 50,
                                                "disp"    : False,}
                                    )

        else:
            opt_results = minimize( fun = self.lagrangian,
                                    x0  = self.v0, 
                                    jac = self.gradient,
                                    hess = self.hessian,
                                    method = opt_method,
                                    tol    = 1e-7,
                                    options = {"maxiter" : 50,
                                                "disp"    : False, }
                                    )

        if opt_results.success == False:
            raise ValueError("Optimization was unsucessful, try a different intitial guess")
        else:
            print("Optimization Successful")

        self.finalize_energy()
        self.generate_grid()

        # if debug=True:
        #     self.density_accuracy()

        self.v = opt_results.x

    def diagonalize_with_guess(self, v):
        """
        Diagonalize Fock matrix with additional external potential
        """
        vks_a = contract("ijk,k->ij", self.S3, v[:self.nbf]) + self.guess_a
        fock_a = self.V + self.T + vks_a 
        self.Ca, self.Coca, self.Da, self.eigvecs_a = self.diagonalize( fock_a, self.nalpha )

        if self.ref == "UHF" or self.ref == "UKS":
            vks_b = contract("ijk,k->ij", self.S3, v[self.nbf:]) + self.guess_b
            fock_b = self.V + self.T + vks_b        
            self.Cb, self.Cocb, self.Db, self.eigvecs_b = self.diagonalize( fock_b, self.nbeta )
        else:
            self.Cb, self.Coca, self.Db, self.eigvecs_b = self.Ca.copy(), self.Coca.copy(), self.Da.copy(), self.eigvecs_a.copy()

    def lagrangian(self, v):
        """
        Lagrangian to be minimized wrt external potential
        Equation (5) of main reference
        """

        self.diagonalize_with_guess(v)
        self.grad_a = contract('ij,ijt->t', (self.Da - self.nt[0]), self.S3)
        self.grad_b = contract('ij,ijt->t', (self.Db - self.nt[1]), self.S3) 

        kinetic     =   contract('ij,ji', self.T, (self.Da + self.Db))
        potential   =   contract('ij,ji', self.V + self.guess_a, self.Da - self.nt[0])
        potential  +=   contract('ij,ji', self.V + self.guess_b, self.Db - self.nt[1])
        optimizing  =   contract('i,i'  , v[:self.naux], self.grad_a)
        optimizing +=   contract('i,i'  , v[self.naux:], self.grad_b)

        if self.debug == True:
            print(f"Kinetic: {kinetic:6.4f} | Potential {np.abs(potential):6.4e} | From Optimization {np.abs(optimizing):6.4e}")
        L = kinetic + potential + optimizing

        reg = 0.0
        if self.reg > 0:
            pass

        return - L - reg

    def gradient(self, v):
        """
        Calculates gradient wrt target density
        Equation (11) of main reference
        """
        self.diagonalize_with_guess(v)
        self.grad_a = contract('ij,ijt->t', (self.Da - self.nt[0]), self.S3)
        self.grad_b = contract('ij,ijt->t', (self.Db - self.nt[1]), self.S3) 
        self.grad   = np.concatenate(( self.grad_a, self.grad_b ))

        return -self.grad

    def hessian(self, v):
        """
        Calculates gradient wrt target density
        Equation (13) of main reference
        """

        self.diagonalize_with_guess(v)

        na, nb = self.nalpha, self.nbeta

        eigs_diff_a = self.eigvecs_a[:na, None] - self.eigvecs_a[None, na:]
        C3a = contract('mi,va,mvt->iat', self.Ca[:,:na], self.Ca[:,nb:], self.S3)
        Ha = 2 * contract('iau,iat,ia->ut', C3a, C3a, eigs_diff_a**-1)

        if self.ref == "UHF" or self.ref == "UKS":
            eigs_diff_b = self.eigvecs_b[:nb, None] - self.eigvecs_b[None, nb:]
            C3b = contract('mi,va,mvt->iat', self.Cb[:,:nb], self.Cb[:,nb:], self.S3)
            Hb = 2 * contract('iau,iat,ia->ut', C3b, C3b, eigs_diff_b**-1)
        else:
            Hb = Ha.copy()

        Hs = np.block( 
                        [[Ha, np.zeros((self.naux, self.naux))],
                            [np.zeros((self.naux, self.naux)), Hb]] 
                        )

        if self.reg > 0.0:
            pass

        return - Hs
