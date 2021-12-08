"""
Provides interface n2v interface to Psi4
"""

import numpy as np
import psi4
from opt_einsum import contract

from .engine import Engine
from ..grid import Psi4Grider

class Psi4Engine(Engine):

    # def __init__(self):
    #     pass

    def set_system(self, molecule, basis, ref='1', pbs='same'):
        """
        Initializes geometry and basis infromation
        """
        self.mol = molecule
        
        #Assert units are in bohr
        # units = self.mol.to_schema(dtype='psi4')['units']
        # if units != "Bohr":
        #     raise ValueError("Units need to be set in Bohr")
        self.basis_str = basis
        self.ref = ref 
        self.pbs = pbs
        self.pbs_str   = basis if pbs == 'same' else pbs

        self.nalpha = None
        self.nbeta = None

    def initialize(self):
        """
        Initializes basic objects required for the Psi4Engine
        """
        self.basis = psi4.core.BasisSet.build( self.mol, key='BASIS', target=self.basis_str)
        self.pbs   = psi4.core.BasisSet.build( self.mol, key='BASIS', target=self.pbs_str)

        self.nbf   = self.basis.nbf()
        self.npbs  = self.pbs.nbf()

        self.mints = psi4.core.MintsHelper( self.basis )
        self.jk    = self.generate_jk()

        self.grid = Psi4Grider(self.mol, self.basis, self.ref)

    def get_T(self):
        """Kinetic Potential in ao basis"""
        return np.array( self.mints.ao_kinetic() )

    def get_Tpbas(self):
        """Kinetic Potential in pbs"""
        return np.array( self.mints.ao_kinetic(self.pbs, self.pbs) )

    def get_V(self):
        """External potential in ao basis"""
        return np.array( self.mints.ao_potential() )

    def get_A(self):
        """Inverse squared root of S matrix"""
        A = self.mints.ao_overlap()
        A.power( -0.5, 1e-16 )
        return np.array( A )

    def get_S(self):
        """Overlap matrix of ao basis"""
        return np.array( self.mints.ao_overlap() )

    def get_S3(self):
        return np.array( self.mints.ao_3coverlap(self.basis,self.basis,self.pbs) )

    def get_S4(self):
        """
        Calculates four overlap integral with Density Fitting method.
        S4_{ijkl} = \int dr \phi_i(r)*\phi_j(r)*\phi_k(r)*\phi_l(r)
        Parameters
        ----------
        wfn: psi4.core.Wavefunction
            Wavefunction object of moleculep
        Return
        ------
        S4
        """

        print(f"4-AO-Overlap tensor will take about {self.nbf **4 / 8 * 1e-9:f} GB.")

        aux_basis = psi4.core.BasisSet.build(self.mol, "DF_BASIS_SCF", "", "JKFIT", self.basis_str)
        S_Pmn = np.squeeze(self.mints.ao_3coverlap(aux_basis, self.basis, self.basis ))
        S_PQ = np.array(self.mints.ao_overlap(aux_basis, aux_basis))
        S_PQinv = np.linalg.pinv(S_PQ, rcond=1e-9)
        S4 = contract('Pmn,PQ,Qrs->mnrs', S_Pmn, S_PQinv, S_Pmn)
        return S4


    def generate_jk(self, gen_K=False):
        """
        Creates jk object for generation of Coulomb and Exchange matrices
        1.0e9 B -> 1.0 GB
        """
        jk = psi4.core.JK.build(self.basis)
        memory = int(jk.memory_estimate() * 1.1)
        jk.set_memory(int(memory)) 
        jk.set_do_K(gen_K)
        jk.initialize()

        return jk

    def compute_hartree(self, Cocc_a, Cocc_b):
        """
        Generates Coulomb and Exchange matrices from occupied orbitals
        """
        Cocc_a = psi4.core.Matrix.from_array(Cocc_a)
        Cocc_b = psi4.core.Matrix.from_array(Cocc_b)
        self.jk.C_left_add(Cocc_a)
        self.jk.C_left_add(Cocc_b) 
        self.jk.compute()
        self.jk.C_clear()
        J = (np.array(self.jk.J()[0]), np.array(self.jk.J()[1]))
        return J


 