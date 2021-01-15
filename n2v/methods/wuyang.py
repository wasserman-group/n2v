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

    Attributes:
    -----------
    lambda_rgl: {None, float}. If float, lambda-regularization is added with lambda=lambda_rgl.
    """

    lambda_rgl = None  # If add lambda regularization


    def wuyang(self, opt_method, opt_max_iter, opt_tol):
        """
        Calls scipy minimizer to minimize lagrangian. 
        """

        if opt_method == 'bfgs':
            opt_results = minimize( fun = self.lagrangian,
                                    x0  = self.v0, 
                                    jac = self.gradient,
                                    method  = opt_method,
                                    tol     = opt_tol,
                                    options = {"maxiter" : opt_max_iter,
                                                "disp"    : False,}
                                    )

        else:
            opt_results = minimize( fun = self.lagrangian,
                                    x0  = self.v0, 
                                    jac = self.gradient,
                                    hess = self.hessian,
                                    method = opt_method,
                                    tol    = opt_tol,
                                    options = {"maxiter"  : opt_max_iter,
                                                "disp"    : False, }
                                    )

        if opt_results.success == False:
            raise ValueError("Optimization was unsucessful, try a different intitial guess")
        else:
            print("Optimization Successful")
            self.v_opt = opt_results.x
            self.opt_info = opt_results

        self.finalize_energy()

        # if debug=True:
        #     self.density_accuracy()

    def diagonalize_with_guess(self, v):
        """
        Diagonalize Fock matrix with additional external potential
        """
        vks_a = contract("ijk,k->ij", self.S3, v[:self.nbf]) + self.va
        fock_a = self.V + self.T + vks_a 
        self.Ca, self.Coca, self.Da, self.eigvecs_a = self.diagonalize( fock_a, self.nalpha )

        if self.ref == 1:
            self.Cb, self.Coca, self.Db, self.eigvecs_b = self.Ca.copy(), self.Coca.copy(), self.Da.copy(), self.eigvecs_a.copy()
        else:
            vks_b = contract("ijk,k->ij", self.S3, v[self.nbf:]) + self.vb
            fock_b = self.V + self.T + vks_b        
            self.Cb, self.Cocb, self.Db, self.eigvecs_b = self.diagonalize( fock_b, self.nbeta )

    def lagrangian(self, v):
        """
        Lagrangian to be minimized wrt external potential
        Equation (5) of main reference
        """

        self.diagonalize_with_guess(v)
        self.grad_a = contract('ij,ijt->t', (self.Da - self.nt[0]), self.S3)
        self.grad_b = contract('ij,ijt->t', (self.Db - self.nt[1]), self.S3) 

        kinetic     =   np.sum(self.T * (self.Da))
        potential   =   np.sum((self.V + self.va) * (self.Da - self.nt[0]))
        optimizing  =   np.sum(v[:self.npbs] * self.grad_a)

        if self.ref == 1:
            L = 2 * (kinetic + potential + optimizing)

        else:
            kinetic    +=   np.sum(self.T * (self.Db))
            potential  +=   np.sum((self.V + self.vb) * (self.Db - self.nt[1]))
            optimizing +=   np.sum(v[self.npbs:] * self.grad_b)
            L = kinetic + potential + optimizing

        if False:
            print(f"Kinetic: {kinetic:6.4f} | Potential: {np.abs(potential):6.4e} | From Optimization: {np.abs(optimizing):6.4e}")

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

        if self.ref == 1:
            self.grad   = self.grad_a
        else:
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
        C3a = contract('mi,va,mvt->iat', self.Ca[:,:na], self.Ca[:,na:], self.S3)
        Ha = 2 * contract('iau,iat,ia->ut', C3a, C3a, eigs_diff_a**-1)

        if self. ref == 1:
            Hs = Ha

        else:
            eigs_diff_b = self.eigvecs_b[:nb, None] - self.eigvecs_b[None, nb:]
            C3b = contract('mi,va,mvt->iat', self.Cb[:,:nb], self.Cb[:,nb:], self.S3)
            Hb = 2 * contract('iau,iat,ia->ut', C3b, C3b, eigs_diff_b**-1)
            Hs = np.block( 
                            [[Ha,                               np.zeros((self.npbs, self.npbs))],
                            [np.zeros((self.npbs, self.npbs)), Hb                              ]]
                        )

        if self.reg > 0.0:
            pass

        return - Hs

    #TODO Regularization
