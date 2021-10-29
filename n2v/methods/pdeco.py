"""
pdeco.py

Functions associated with PDE-Constrained Optimization.
"""

import numpy as np
from opt_einsum import contract
from scipy.optimize import minimize
try:
    from psi4.core import BasisSet as psi4_basiset
    from psi4.core import MintsHelper as psi4_mintshelper
except:
    pass

class PDECO():
    """
    Performs Optimization as in: 10.1063/1.1535422 - Qin Wu + Weitao Yang

    Attributes:
    -----------
    lambda_rgl: {None, float}. If float, lambda-regularization is added with lambda=lambda_rgl.
    """

    regul_norm = None  # Regularization norm: ||v||^2
    lambda_reg = None  # Regularization constant

    def pdeco(self, opt_max_iter, reg=None, gtol=1e-3,
              opt_method='L-BFGS-B', opt=None):
        """
        Calls scipy minimizer to minimize lagrangian. 
        """
        self.lambda_reg = reg

        self.lambda_reg = reg
        if opt is None:
            opt = {"disp": False}
        opt['maxiter'] = opt_max_iter
        opt['gtol'] = gtol

        # Initialization for D and C
        self._diagonalize_with_potential_pbs(self.v_pbs)

        if self.S4 is None:
            self.S4 = self.fouroverlap()

        if opt_method.lower() == 'bfgs' or opt_method.lower() == 'l-bfgs-b':
            opt_results = minimize( fun = self.lagrangian_pbeco,
                                    x0  = self.v_pbs, 
                                    jac = self.gradient_pbeco,
                                    method  = opt_method,
                                    options = opt
                                    )
        else:
            raise ValueError(F'{opt_method} is not supported. Only BFGS '
                             F'and L-BFGS is supported.')

        if opt_results.success == False:
            self.v_pbs = opt_results.x
            self.opt_info = opt_results
            raise ValueError("Optimization was unsucessful (|grad|=%.2e) within %i iterations, "
                             "try a different initial guess. %s"% (np.linalg.norm(opt_results.jac), opt_results.nit, opt_results.message)
                             )
        else:
            print(f"Optimization Successful within {opt_results.nit} iterations! |grad|={np.linalg.norm(opt_results.jac):.2e}." )
            self.v_pbs = opt_results.x
            self.opt_info = opt_results


    def fouroverlap(self, wfn=None):
        """
        Calculates four overlap integral with Density Fitting method.
        S4_{ijkl} = \int dr \phi_i(r)*\phi_j(r)*\phi_k(r)*\phi_l(r)

        Parameters
        ----------
        wfn: psi4.core.Wavefunction
            Wavefunction object of molecule

        Return
        ------
        S4
        """
        if wfn is None:
            wfn = self.wfn

        print(f"4-AO-Overlap tensor will take about {self.nbf **4 / 8 * 1e-9:f} GB.")

        mints = psi4_mintshelper( self.basis )

        aux_basis = psi4_basiset.build(wfn.molecule(), "DF_BASIS_SCF", "",
                                     "JKFIT", wfn.basisset().name())
        S_Pmn = np.squeeze(mints.ao_3coverlap(aux_basis, wfn.basisset(),
                                              wfn.basisset()))
        S_PQ = np.array(mints.ao_overlap(aux_basis, aux_basis))

        S_PQinv = np.linalg.pinv(S_PQ, rcond=1e-9)

        S4 = np.einsum('Pmn,PQ,Qrs->mnrs', S_Pmn, S_PQinv, S_Pmn, optimize=True)
        return S4

    def lagrangian_pbeco(self, v):
        """
        Lagrangian to be minimized wrt external potential
        Equation (5) of main reference
        """

        # If v is not updated, will not re-calculate.
        if not np.allclose(v, self.v_pbs, atol=1e-15):
            self._diagonalize_with_potential_pbs(v)

        # self._diagonalize_with_potential_pbs(v)

        if self.ref == 1:
            L = 4 * contract("ijkl,ij,kl", self.S4, self.Da - self.Dt[0], self.Da- self.Dt[0])
        else:
            L = contract("ijkl,ij,kl", self.S4, self.Da+self.Db-self.Dt[0]-self.Dt[1], self.Da+self.Db-self.Dt[0]-self.Dt[1])
        # Add lambda-regularization
        if self.lambda_reg is not None:
            T = self.T_pbs
            if self.ref == 1:
                norm = 2 * (v[:self.npbs] @ T @ v[:self.npbs])
            else:
                norm = (v[self.npbs:] @ T @ v[self.npbs:]) + (v[:self.npbs] @ T @ v[:self.npbs])

            L += norm * self.lambda_reg
            self.regul_norm = norm
        return L

    def gradient_pbeco(self, v):
        """
        Calculates gradient wrt target density
        Equation (11) of main reference
        """
        # If v is not updated, will not re-calculate.
        if not np.allclose(v, self.v_pbs, atol=1e-15):
            self._diagonalize_with_potential_pbs(v)

        if self.ref == 1:
            grad_temp = np.zeros((self.nbf, self.nbf))
            g = 8 * contract("ijkl,ij,km->lm", self.S4, 2 * (self.Dt[0] - self.Da), self.Coca)  # shape (ao, mo)
            u = 0.5 * contract("lm,lm->m", self.Coca, g)  # shape (mo, )
            g -= 2 * contract('m,ij,jm->im', u, self.S2, self.Coca) # shape (ao, mo)
            for idx in range(self.nalpha):
                LHS = self.Fock - self.S2 * self.eigvecs_a[idx]
                p_i = np.linalg.solve(LHS, g[:, idx])
                # Gram–Schmidt rotation
                p_i -= np.sum(p_i * np.dot(self.S2, self.Coca[:,idx])) * self.Coca[:,idx]
                assert np.allclose([np.sum(p_i * (self.S2 @ self.Coca[:,idx])), np.linalg.norm(np.dot(LHS,p_i)-g[:, idx]), np.sum(g[:, idx]*self.Coca[:,idx])], 0, atol=1e-4)
                grad_temp += p_i[:, np.newaxis] * self.Coca[:,idx]

            self.grad = contract("ij,ijk->k", grad_temp, self.S3)
        else:
            grad_temp_a = np.zeros((self.nbf, self.nbf))
            g_a = 4 * contract("ijkl,ij,km->lm", self.S4, (self.Dt[0] - self.Da) + (self.Dt[1] - self.Db), self.Coca)  # shape (ao, mo)
            u_a = 0.5 * contract("lm,lm->m", self.Coca, g_a)  # shape (mo, )
            g_a -= 2 * contract('m,ij,jm->im', u_a, self.S2, self.Coca) # shape (ao, mo)
            for idx in range(self.nalpha):
                LHS = self.Fock[0] - self.S2 * self.eigvecs_a[idx]
                p_i = np.linalg.solve(LHS, g_a[:, idx])
                # Gram–Schmidt rotation
                p_i -= np.sum(p_i * np.dot(self.S2, self.Coca[:,idx])) * self.Coca[:,idx]
                assert np.allclose([np.sum(p_i * (self.S2 @ self.Coca[:,idx])), np.linalg.norm(np.dot(LHS,p_i)-g_a[:, idx]), np.sum(g_a[:, idx]*self.Coca[:,idx])], 0, atol=1e-4)
                grad_temp_a += p_i[:, np.newaxis] * self.Coca[:,idx]

            grad_temp_b = np.zeros((self.nbf, self.nbf))
            g_b = 4 * contract("ijkl,ij,km->lm", self.S4, (self.Dt[0] - self.Da) + (self.Dt[1] - self.Db), self.Cocb)  # shape (ao, mo)
            u_b = 0.5 * contract("lm,lm->m", self.Cocb, g_b)  # shape (mo, )
            g_b -= 2 * contract('m,ij,jm->im', u_b, self.S2, self.Cocb) # shape (ao, mo)
            for idx in range(self.nbeta):
                LHS = self.Fock[1] - self.S2 * self.eigvecs_b[idx]
                p_i = np.linalg.solve(LHS, g_b[:, idx])
                # Gram–Schmidt rotation
                p_i -= np.sum(p_i * np.dot(self.S2, self.Cocb[:,idx])) * self.Cocb[:,idx]
                assert np.allclose([np.sum(p_i * (self.S2 @ self.Cocb[:,idx])), np.linalg.norm(np.dot(LHS,p_i)-g_b[:, idx]), np.sum(g_b[:, idx]*self.Cocb[:,idx])], 0, atol=1e-4)
                grad_temp_b += p_i[:, np.newaxis] * self.Cocb[:,idx]

            self.grad = np.concatenate((contract("ij,ijk->k", grad_temp_a, self.S3), contract("ij,ijk->k", grad_temp_b, self.S3)))

        if self.lambda_reg is not None:
            T = self.T_pbs
            if self.ref == 1:
                rgl_vector = 4 * self.lambda_reg*np.dot(T, v[:self.npbs])
                self.grad += rgl_vector
            else:
                self.grad[:self.npbs] += 2 * self.lambda_reg*np.dot(T, v[:self.npbs])
                self.grad[self.npbs:] += 2 * self.lambda_reg*np.dot(T, v[self.npbs:])
        return self.grad

    def find_regularization_constant_pdeco(self, opt_max_iter, opt_method="L-BFGS-B", gtol=1e-3,
                                     opt=None, lambda_list=None):
        """
        Finding regularization constant lambda.

        Note: it is recommend to set a specific convergence criteria by opt or tol,
                in order to control the same convergence
                for different lambda value.

        After the calculation is done, one can plot the returns to select a good lambda.

        Parameters:
        -----------
        guide_potential_components: a list of string
            the components for guide potential v_pbs.
            see Inverter.generate_components() for details.

        opt_method: string default: "trust-krylov"
            opt_methods available in scipy.optimize.minimize

        tol: float
            Tolerance for termination. See scipy.optimize.minimize for details.

        opt: dictionary, optional
            if given:
                scipy.optimize.minimize(method=opt_method, options=opt)

        lambda_list: np.ndarray, optional
            A array of lambda to search; otherwise, it will be 10 ** np.linspace(-1, -7, 7).

        Returns:
        --------
        lambda_list: np.ndarray
            A array of lambda searched.

        P_list: np.ndarray
            The value defined by [Bulat, Heaton-Burgess, Cohen, Yang 2007] eqn (21).
            Corresponding to lambda in lambda_list.

        error_list: np.ndarray
            The Ts value for each lambda.


        """

        error_list = []
        v_norm_list = []


        if lambda_list is None:
            lambda_list = 10 ** np.linspace(-3, -9, 7)

        if opt is None:
            opt = {"disp"    : False}
        opt['maxiter'] = opt_max_iter
        opt['gtol'] = gtol

        self.lambda_reg = None
        # Initial calculation with no regularization
        # Initialization for D and C
        self._diagonalize_with_potential_pbs(self.v_pbs)
        if opt_method.lower() == 'bfgs' or opt_method.lower() == 'l-bfgs-b':
            initial_result = minimize(fun=self.lagrangian_pbeco,
                                   x0=self.v_pbs,
                                   jac=self.gradient_pbeco,
                                   method=opt_method,
                                   options=opt
                                   )
        else:
            raise ValueError(F'{opt_method} is not supported. Only BFGS '
                             F'and L-BFGS is supported.')

        if initial_result.success == False:
            raise ValueError("Optimization was unsucessful (|grad|=%.2e) within %i iterations, "
                             "try a different intitial guess"% (np.linalg.norm(initial_result.jac), initial_result.nit)
                             + initial_result.message)
        else:
            error0 = -initial_result.fun
            initial_v_pbs = initial_result.x  # This is used as the initial guess for with regularization calculation.

        for reg in lambda_list:
            self.lambda_reg = reg

            if opt_method.lower() == 'bfgs' or opt_method.lower() == 'l-bfgs-b':
                opt_results = minimize(fun=self.lagrangian_pbeco,
                                       x0=initial_v_pbs,
                                       jac=self.gradient_pbeco,
                                       method=opt_method,
                                       options=opt
                                       )

            else:
                raise ValueError(F'{opt_method} is not supported. Only BFGS '
                                 F'and L-BFGS is supported.')


            v_norm_list.append(self.regul_norm)
            error_list.append(opt_results.fun - self.lambda_reg * self.regul_norm)

        P_list = lambda_list * np.array(v_norm_list) / (np.array(error_list) - error0)

        return lambda_list, P_list, np.array(error_list)
