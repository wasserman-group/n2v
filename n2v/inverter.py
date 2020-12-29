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
    def __init__(self, wfn, aux_str="same", debug=False):
        """
        Handles Inversion
        
        Parameters
        ----------
        wfn : psi4.core.{RHF, UHF, RKS, UKS, Wavefunction, CCWavefuncion...}
            Psi4 wavefunction object
        mol : psi4.core.Molecule
            Psi4 molecule object
        basis : psi4.core.BasisSet
            Psi4 basis set object
        basis_str : str
            Basis set
        nbf : int
            Number of basis functions for main calculation
        nalpha : int
            Number of alpha electrons
        nbeta : int 
            Number of beta electrons
        ref : {1,2}
            Reference calculation
            1 -> Restricted
            2 -> Unrestricted
        nt : List
            List of psi4.core.Matrix for target densities
        ct : List
            List of psi4.core.Matrix for occupied orbitals
        aux : psi4.core.BasisSet
            Auxiliary basis set for calculation of potential
        v0  : np.ndarray
            Initial zero guess for optimizer
        """
        self.wfn       = wfn
        self.mol       = wfn.molecule()
        self.basis     = wfn.basisset()
        self.basis_str = wfn.basisset().name()
        self.nbf       = wfn.basisset().nbf()
        self.nalpha    = wfn.nalpha()
        self.nbeta     = wfn.nbeta()
        self.ref       = 1 if psi4.core.get_global_option("REFERENCE") == "RHF" or \
                              psi4.core.get_global_option("REFERENCE") == "RKS" else 2
        self.nt        = [wfn.Da().np, wfn.Db().np]
        self.ct        = [wfn.Ca_subset("AO", "OCC"), wfn.Cb_subset("AO", "OCC")]
        self.aux       = self.basis if aux_str is "same" \
                                    else psi4.core.BasisSet.build( self.mol, key='BASIS', target=self.aux_str)
        self.naux      = self.aux.nbf()
        self.v0        = np.zeros( (self.naux) ) if self.ref == 1 \
                                                 else np.zeros( 2 * self.naux )
        self.generate_mints_matrices()

    #------------->  Basics:

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

    def generate_jk(self, gen_K=False, memory=2.50e9):
        """
        Creates jk object for generation of Coulomb and Exchange matrices
        1.0e9 B -> 1.0 GB
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
        K = []

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

    def invert(self, method="wuyang", opt_method='trust-krylov', potential_components = ["fermi_amaldi", "svwn"], reg=0.0):
        """
        Handler to all available inversion methods
        """

        self.reg = reg
        self.generate_components(v_components)

        if method.lower() == "wuyang":
            self.wuyang(opt_method)
        elif method.lower() == "pde":
            pass
        elif method.lower() == "mrks":
            pass
        else:
            raise ValueError(f"Inversion method not available. Try: {['wuyang', 'pde', 'mrks']}")

    def generate_components(self, potential_components):
        """
        Generates exact
        """

        self.v = np.zeros_like(self.T)
        self.v = np.zeros_like(self.T)

        if "fermi_amaldi" in guess:
            N = self.nalpha + self.nbeta
            J, _ = self.form_jk( self.ct[0], self.ct[1] )
            self.Hartree_a, self.Hartree_b = J[0], J[1]
            v_fa = (-1/N) * (J[0] + J[1])

            self.v += v_fa
            self.v += v_fa

        if "svwn" in guess or "pbe" in guess:

            if "svwn" in guess:
                _, wfn_guess = psi4.energy( "svwn"+"/"+self.basis_str, molecule=self.mol , return_wfn = True)
            else:
                _, wfn_guess = psi4.energy( "pbe"+"/"+self.basis_str, molecule=self.mol , return_wfn = True)
            #Get density-drivenless vxc
            if self.ref == 1:
                ntarget = psi4.core.Matrix.from_array( [ self.nt[0] + self.nt[1] ] )
                wfn_guess.V_potential().set_D( [ntarget] )
                va_target = psi4.core.Matrix( self.nbf, self.nbf )
                wfn_guess.V_potential().compute_V([va_target])
                self.guess_a += va_target.np
                self.guess_b += va_target.np
            elif self.ref == 2:
                na_target = psi4.core.Matrix.from_array( self.nt[0] )
                nb_target = psi4.core.Matrix.from_array( self.nt[1] )
                wfn_guess.V_potential().set_D( [na_target, nb_target] )
                va_target = psi4.core.Matrix( self.nbf, self.nbf )
                vb_target = psi4.core.Matrix( self.nbf, self.nbf )
                wfn_guess.V_potential().compute_V([va_target, vb_target])
                self.guess_a += va_target.np
                self.guess_b += vb_target.np

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
        energies = {"One-Electron Energy" : energy_kinetic + energy_external,
                    "Two-Electron Energy" : energy_hartree_a + energy_hartree_b,
                    "XC"                  : energy_ks,
                    "Total Energy"        : energy_kinetic   + energy_external  + \
                                            energy_hartree_a + energy_hartree_b + \
                                            energy_ks }
        self.energy   = energies["Total Energy"] 
        self.energies = energies

        print(f"Final Energies: {self.energies}")