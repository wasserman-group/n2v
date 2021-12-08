"""
Provides interface n2v interface to PySCF
"""

import numpy as np
import scipy
from opt_einsum import contract
from gbasis.wrappers import from_pyscf

import sys

try: 
    from pyscf import gto, scf, lib, dft
except:
    ImportError("Pyscf is not installed. Try installing it before using the Engine")

from .engine import Engine

class PySCFEngine(Engine):

    def set_system(self, molecule, basis, ref=1, pbs='same'):
        """
        Stores basic information from a PySCF calculation

        Parameters
        ----------
        
        mol: pyscf.gto.mole.Mole
            Pyscf molecule object
        basis: str
            Basis set for calculation
        ref: {1,2}
            1 -> Restricted 
            2 -> Unrestricted
        pbs: str. default: "same" (as calculation)
            Basis set for expressing inverted potential
        """
        self.mol = molecule
        self.basis_str = basis
        self.pbs_str = pbs
        self.ref = ref

        if self.pbs_str != 'same': # Builds additional mole for secondary basis set
            self.mol_pbs = gto.Mole()
            self.mol_pbs.atom = self.mol.atom
            self.mol_pbs.basis = self.pbs_str
            self.mol_pbs.build()

        self.nalpha = self.mol.nelec[0]
        self.nbeta = self.mol.nelec[1]

    def initialize(self):
        """
        Initializes different components for calculation.
        """
        self.nbf = self.mol.nao_nr()
        if self.pbs_str == 'same':
            self.npbs = self.nbf
        else:
            self.npbs = self.mol_pbs.nao_nr()

    def get_T(self):
        """
        Generates Kinetic Operator in AO basis.
        
        Returns
        -------
        T: np.ndarray. Shape: (nbf, nbf)
        """
        return self.mol.intor('int1e_kin')

    def get_Tpbas(self):
        """
        Generates Kinetic Operator in AO basis for additional basis. 

        Returns
        -------
        T_pbas: np.ndarray. Shape: (nbf, nbf)
        """
        return self.mol_pbs.intor('int1e_kin')

    def get_V(self):
        """
        Generates External Potential in AO basis

        Returns
        -------
        V: np.ndarray. Shape: (nbf, nbf)
        """
        return self.mol.intor('int1e_nuc')

    def get_A(self):
        """
        Generates S^(-0.5)

        Returns
        -------
        A: np.ndarray. Shape: (nbf, nbf)
        """
        A = self.mol.intor('int1e_ovlp')
        A = scipy.linalg.fractional_matrix_power(A, -.5)
        return A

    def get_S(self):
        """
        Builds Overlap matrix of AO basis

        Returns
        -------
        S: np.ndarray. Shape: (nbf, nbf)
        """
        return self.mol.intor('int1e_ovlp')

    def get_S3(self):
        """
        Builds 3 Overlap Matrix. 
        Manually built since Pyscf does not support it. 

        Returns
        -------
        S3: np.ndarray. Shape: (nbf, nbf, nbf or npbs)
            Third dimension depends on wether an additional basis is used. 
        """

        grid = dft.gen_grid.Grids(self.mol)
        grid.build()
        bs1 = dft.numint.eval_ao(self.mol, grid.coords)

        if self.pbs_str == 'same':
            S3 = contract('ij, ik, il, i -> jkl', bs1, bs1, bs1, grid.weights)
            bs1 = None

        else:
            bs2 = dft.numint.eval_ao(self.mol_pbs, grid.coords)
            S3 = contract('ij, ik, il, i -> jkl', bs1, bs1, bs2, grid.weights)
            bs1, bs2 = None, None

        return S3

    def get_S4(self):
        sys.exit("S4 Not implemented yet")

    def compute_hartree(self, ca, cb=None):
        """
        Computes Hartree Operator in AO basis

        Parameters
        ----------
        ca: np.ndarray
            Occupied Orbitals in AO basis
        cb: np.ndarray
            if ref == 2, cb -> Beta Occupied Orbitals in AO basis
        """

        da = (ca @ ca.T)
        db = (cb @ cb.T)
        mf = dft.uks.UKS(self.mol)
        J = mf.get_j(dm=[da, db])

        return J
    
    
    