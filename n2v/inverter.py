"""
inverter.py
Density-to-potential inversion module

Handles the primary functions
"""

import numpy as np
from scipy.optimize import minimize
from opt_einsum import contract

import psi4
psi4.core.be_quiet()

from .methods._wuyang import WuYang


class Inverter(WuYang):
    def __init__(self, mol, basis_str, aux_str="same", debug=False):
        self.basis_str = basis_str
        self.aux_str   = aux_str
        self.mol       = mol
        self.build_basis()
        self.generate_mints_matrices()
        self.generate_jk()

        #Plotting Grid
        # self.grid = Grider()
        #Inversion
        self.v0 = np.zeros_like( 2 * self.naux )
        self.reg = 0.0
        #Anything else
        self.debug = debug


    #------------->  Basics:

    def build_basis(self):
        """
        Build basis set object and auxiliary basis set object
        """

        basis = psi4.core.BasisSet.build( self.mol, key='BASIS', target=self.basis_str)
        self.basis = basis
        self.nbf   = self.basis.nbf()

        if self.aux_str != "same":
            aux_basis = psi4.core.BasisSet.build( self.mol, key='Basis', target=self.aux_str)
            self.aux = aux_basis
            self.naux = aux_basis.nbf()
        else:
            self.aux  = self.basis
            self.naux = self.nbf

    def generate_mints_matrices(self):
        """
        Generates matrices that are methods of a mints object
        """

        mints = psi4.core.MintsHelper( self.basis )

        #Overlap Matrices
        self.S2 = mints.ao_overlap().np
        A = mints.ao_overlap()
        A.power( -0.5, 1e-16 )
        self.A = A
        self.S3 = np.squeeze(mints.ao_3coverlap(self.basis,self.basis,self.aux))
        self.jk = None 

        #Core Matrices
        self.T = mints.ao_kinetic().np.copy()
        self.V = mints.ao_potential().np.copy()

    def generate_jk(self, gen_K=True, memory=2.50e9):
        """
        Creates jk object for generation of Coulomb and Exchange matrices
        2.5e9 B -> 2.5 GB
        """
        jk = psi4.core.JK.build(self.basis)
        jk.set_memory(int(memory)) 
        jk.set_do_K(gen_K)
        jk.initialize()
        self.jk = jk

    def form_jk(self, Cocc_a, Cocc_b):
        """
        Generates Coulomb and Exchange matrices from occupied orbitals
        """

        self.jk.C_left_add(Cocc_a)
        self.jk.C_left_add(Cocc_b)
        self.jk.compute()
        self.jk.C_clear()

        J = [self.jk.J()[0].np, self.jk.J()[1].np]
        K = [self.jk.K()[0].np, self.jk.K()[1].np]

        return J, K

    def diagonalize(self, matrix, ndocc):
        matrix = psi4.core.Matrix.from_array( matrix )
        Fp = psi4.core.triplet(self.part.A, matrix, self.part.A, True, False, True)
        nbf = self.part.A.shape[0]
        Cp = psi4.core.Matrix(nbf, nbf)
        eigvecs = psi4.core.Vector(nbf)
        Fp.diagonalize(Cp, eigvecs, psi4.core.DiagonalizeOrder.Ascending)
        C = psi4.core.doublet(self.part.A, Cp, False, False)
        Cocc = psi4.core.Matrix(nbf, ndocc)
        Cocc.np[:] = C.np[:, :ndocc]
        D = psi4.core.doublet(Cocc, Cocc, False, True)

        return C.np, Cocc.np, D.np, eigvecs.np

    #------------->  Inversion:

    def invert(self, wfn, method, opt_method='bfgs', guess=["fermi_amaldi"]):
        """
        Handler to all available inversion methods
        """

        self.nt = [wfn.Da().np, wfn.Db().np]
        self.ct = [wfn.Ca_subset("AO", "OCC"), wfn.Cb_subset("AO", "OCC")]
        self.initial_guess(guess)

        if method.lower() == "wuyang":
            self.wuyang(opt_method)
        if method.lower() == "pde":
            pass
        if method.lower() == "mrks":
            pass

    def initial_guess(self, guess):

        self.guess_a = np.zeros_like(self.T)
        self.guess_b = np.zeros_like(self.T)

        if "fermi_amaldi" in guess:
            if self.debug == True:
                print("Adding Fermi Amaldi potential to initial guess")

            N = self.mol.nallatom()
            J, _ = self.form_jk( self.ct[0], self.ct[1] )
            v_fa = (-1/N) * (J[0] + J[1])

            self.guess_a += v_fa
            self.guess_b += v_fa

        if "svwn" in guess or "pbe" in guess:
            if self.debug == True:
                print(f"Adding XC potential {method} to initial guess")

            if "svwn" in guess:
                method = guess.index("svwn")
            elif "pbe" in guess:
                method = guess.index("pbe")

            _, wfn_guess = psi4.energy( method+"/"+self.basis_str, molecule=self.mol , return_wfn = True)
            na_target = self.nt[0]
            nb_target = self.nt[1]
            self.nalpha = wfn_guess.nalpha()
            self.nbeta = wfn_guess.nbeta()
            #Get density-drivenless vxc
            wfn_guess.V_potential().set_D( [na_target, nb_target] )
            va_target = psi4.core.Matrix( self.nbf, self.nbf )
            vb_target = psi4.core.Matrix( self.nbf, self.nbf )
            wfn_guess.V_potential().compute_V([va_target, vb_target])

            self.guess_a += va_target
            self.guess_b += vb_target



