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


def Inverter()

    def __init__(self, mol, basis_str, nt, aux="same"):

        self.basis_str = basis_str
        self.aux_str   = aux_str

        self.mol       = mol
        self.nt        = nt

        #Verify reference
        self.ref = psi4.core.get_global_option("REFERENCE")
        if self.ref == "UKS" or self.ref == "UHF" and len(nt) == 1:
            raise ValueError("Reference is set as Unrestricted but target density's dimension == 1")
        if self.ref == "RKS" or self.ref == "RHF" and len(nt) == 2:
            raise ValueError("Reference is set as Restricted but target density's dimension == 2")
        if len(nt == 1):
            nt.append(nt[0])

        #Generate Basis set
        self.build_basis()
        #Generate matrices from Mints
        self.generate_mint_matrices()
        self.generate_core_matrices()
        self.generate_jk()

        #Plotting Grid
        self.grid = Grider()


    #-------------  Basics

    def build_basis(self):
        """
        Build basis set object and auxiliary basis set object
        """

        basis = psi4.core.BasisSet.build( mol, key='BASIS', target=self)
        self.basis = basis
        self.nbf   = self.basis.nbf()

        if self.aux_str is not "same":
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
        self.T = self.mints.ao_kinetic().np.copy()
        self.V = self.mints.ao_potential().np.copy()

        self.mints = mints

    def generate_jk(self, gen_K=True, memory=2.50e9):
        """
        Creates jk object for generation of Coulomb and Exchange matrices
        2.5e9 B -> 2.5 GB
        """
        jk = psi4.core.JK.build(self.basis)
        jk.set_memory(int(memory)) 
        jk.set_do_K(gen_K)
        jk.initialize()

    def form_jk(self, coca, cocb):
        """
        Generates Coulomb and Exchange matrices from occupied orbitals
        """

        self.jk.C_left_add(C_occ_a)
        self.jk.C_left_add(C_occ_b)
        self.jk.compute()
        self.jk.C_clear()

        J = [self.jk.J()[0], self.jk.J()[1]]
        K = [self.jk.K()[0], self.jk.K()[1]]

        return J, K




    #Inversion

    def invert(self, method, guess):

        self.initial_guess(method)


    def initial_guess(self, method):

        self.guess_a = 
        self.guess_b =

        if "fermi_amaldi" in method:


        if "svwn" in method or "pbe" in method:
            if self.debug == True:
                print(f"Adding XC potential {method} to initial guess")

            _, wfn_guess = psi4.energy( method+/+self.basis_str, molecule=self.mol , return_wfn = True)
            na_target = self.nt[0]
            nb_target = self.nt[1]
            #Get density-drivenless vxc
            wfn_guess.V_potential().set_D( [na_target, nb_target] )
            va_target = psi4.core.Matrix( self.nbf, self.nbf )
            vb_target = psi4.core.Matrix( self.nbf, self.nbf )
            wfn_guess.V_potential().compute_V([va_target, vb_target])

            self.guess_a += va_target
            self.guess_b += vb_target













