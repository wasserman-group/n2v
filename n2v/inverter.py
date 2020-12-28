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

from .methods.wuyang import WuYang
from .grider import Grider


class Inverter(WuYang, Grider):
    def __init__(self, mol, basis_str, aux_str="same", debug=False):
        self.basis_str = basis_str
        self.aux_str   = aux_str
        self.mol       = mol
        self.ref       = psi4.core.get_global_option("REFERENCE")
        self.build_basis()
        self.generate_mints_matrices()
        self.generate_jk()
        #Inversion
        self.v0 = np.zeros( (2 * self.naux) )
        self.reg = 0.0
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
        """
        Diagonalizes Fock Matrix
        """
        matrix = psi4.core.Matrix.from_array( matrix )
        Fp = psi4.core.triplet(self.A, matrix, self.A, True, False, True)
        Cp = psi4.core.Matrix(self.nbf, self.nbf)
        eigvecs = psi4.core.Vector(self.nbf)
        Fp.diagonalize(Cp, eigvecs, psi4.core.DiagonalizeOrder.Ascending)
        C = psi4.core.doublet(self.A, Cp, False, False)
        Cocc = psi4.core.Matrix(self.nbf, ndocc)
        Cocc.np[:] = C.np[:, :ndocc]
        D = psi4.core.doublet(Cocc, Cocc, False, True)

        return C.np, Cocc.np, D.np, eigvecs.np

    #------------->  Inversion:

    def invert(self, wfn, method, opt_method='bfgs', guess=["fermi_amaldi"]):
        """
        Handler to all available inversion methods
        """

        self.nalpha, self.nbeta = wfn.nalpha(), wfn.nbeta()
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
        """
        Generates Initial guess for inversion
        """

        self.guess_a = np.zeros_like(self.T)
        self.guess_b = np.zeros_like(self.T)

        if "fermi_amaldi" in guess:
            if self.debug is True:
                print("Adding Fermi Amaldi potential to initial guess")

            N = self.nalpha + self.nbeta
            J, _ = self.form_jk( self.ct[0], self.ct[1] )
            self.Hartree_a, self.Hartree_b = J[0], J[1]
            v_fa = (-1/N) * (J[0] + J[1])

            # print("J target\n", J[0] + J[1])

            self.guess_a += v_fa
            self.guess_b += v_fa

        if "svwn" in guess or "pbe" in guess:
            if "svwn" in guess:
                indx = guess.index("svwn")
                method = guess[indx]
            elif "pbe" in guess:
                indx = guess.index("pbe")
                method = guess[indx]

            if self.debug == True:
                print(f"Adding XC potential to initial guess")

            _, wfn_guess = psi4.energy( method+"/"+self.basis_str, molecule=self.mol , return_wfn = True)
            self.nalpha = wfn_guess.nalpha()
            self.nbeta = wfn_guess.nbeta()
            #Get density-drivenless vxc
            if self.ref == "UKS" or self.ref == "UHF":
                na_target = psi4.core.Matrix.from_array( self.nt[0] )
                nb_target = psi4.core.Matrix.from_array( self.nt[1] )
                wfn_guess.V_potential().set_D( [na_target, nb_target] )
                va_target = psi4.core.Matrix( self.nbf, self.nbf )
                vb_target = psi4.core.Matrix( self.nbf, self.nbf )
                wfn_guess.V_potential().compute_V([va_target, vb_target])
                self.guess_a += va_target.np
                self.guess_b += vb_target.np
            else:
                ntarget = psi4.core.Matrix.from_array( [ self.nt[0] + self.nt[1] ] )
                wfn_guess.V_potential().set_D( [ntarget] )
                v_target = psi4.core.Matrix( self.nbf, self.nbf )
                wfn_guess.V_potential().compute_V([v_target])

                self.guess_a += v_target.np / 2
                self.guess_b += v_target.np / 2

    def generate_grid(self):
        self.get_from_grid()

    def finalize_energy(self):
        """
        Calculates energy contributions
        """

        energy_kinetic    = contract('ij,ij', self.T, (self.Da + self.Db))
        energy_external   = contract('ij,ij', self.V, (self.Da + self.Db))
        energy_hartree_a  = 0.5 * contract('ij,ji', self.Hartree_a + self.Hartree_b, self.Da)
        energy_hartree_b  = 0.5 * contract('ij,ji', self.Hartree_a + self.Hartree_b, self.Db)

        print("WARNING: XC Energy is not yet properly calculated")
        energy_ks = 0.0

        # # alpha = 0.0
        # bucket = get_from_grid(self.part.mol_str, self.part.basis_str, self.Da, self.Db )
        # # energy_exchange_a = -0.5 * alpha * contract('ij,ji', K[0], self.Da)
        # # energy_exchange_b = -0.5 * alpha * contract('ij,ji', K[1], self.Db)
        # energy_ks            =  1.0 * bucket.exc

        energies = {"One-Electron Energy" : energy_kinetic + energy_external,
                    "Two-Electron Energy" : energy_hartree_a + energy_hartree_b,
                    "XC"                  : energy_ks,
                    "Total Energy"        : energy_kinetic   + energy_external  + \
                                            energy_hartree_a + energy_hartree_b + \
                                            energy_ks }
        self.energy   = energies["Total Energy"] 
        self.energies = energies

        print(f"Final Energies: {self.energies}")