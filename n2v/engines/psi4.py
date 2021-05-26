"""
Provides interface n2v interface to Psi4
"""

import numpy as np
import psi4

from .engine import Engine
from ..grid import Psi4Grider

class Psi4Engine(Engine):

    def __init__(self):
        pass

    def initialize(self):
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
        """Inverse Squared root of S matrix"""
        A = self.mints.ao_overlap()
        A.power( -0.5, 1e-16 )
        return np.array( A )
    def get_S(self):
        """Overlap matrix of ao basis"""
        return np.array( self.mints.ao_overlap() )
    def get_S3(self):
        return np.array( self.mints.ao_3coverlap(self.basis,self.basis,self.pbs) )

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
        Computes har
        """
        Cocc_a = psi4.core.Matrix.from_array(Cocc_a)
        Cocc_b = psi4.core.Matrix.from_array(Cocc_b)
        self.jk.C_left_add(Cocc_a)
        self.jk.C_left_add(Cocc_b) 
        self.jk.compute()
        self.jk.C_clear()
        J = (np.array(self.jk.J()[0]), np.array(self.jk.J()[1]))
        return J

