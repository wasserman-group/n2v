"""
Provides interface n2v interface to PySCF
"""

import numpy as np
import scipy
from opt_einsum import contract

from .engine import Engine

try:
    from pyscf import gto, dft
    has_pyscf = True
except ImportError:
    has_pyscf = False

if has_pyscf:
    from ..grid import PySCFGrider
    class PySCFEngine(Engine):
        """
        PySCF Engine
        """

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

            if pbs != 'same': # Builds additional mole for secondary basis set
                self.pbs = gto.Mole()
                self.pbs.atom = self.mol.atom
                self.pbs.basis = self.pbs_str
                self.pbs.build()
            else: 
                self.pbs = None

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
                self.npbs = self.pbs.nao_nr()

            self.grid = PySCFGrider(self.mol, self.pbs)
        
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
            return self.pbs.intor('int1e_kin')

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
                del bs1

            else:
                bs2 = dft.numint.eval_ao(self.pbs, grid.coords)
                S3 = contract('ij, ik, il, i -> jkl', bs1, bs1, bs2, grid.weights)
                del bs1
                del bs2

            return S3

        def get_S4(self):
            """
            Obtains a 4 AO Overlap Matrix using Density Fitting.
            """
            
            grid = dft.gen_grid.Grids(self.mol)
            grid.build()
            bs1 = dft.numint.eval_ao(self.mol, grid.coords)

            # PySCF's DF basis are identified by the suffix "-jk-fit".
            auxbasis = self.mol.basis + '-jk-fit'
            mol_aux = gto.M(atom=self.mol.atom, basis=auxbasis)
            bs2 = dft.numint.eval_ao(mol_aux, grid.coords)

            S_Pmn = contract('ij, ik, il, i -> jkl', bs2, bs1, bs1, grid.weights)
            S_PQ = mol_aux.mol.intor('int1e_ovlp')
            S_PQinv = np.linalg.pinv(S_PQ, rcond=1e-9)
            S4 = contract('Pmn,PQ,Qrs->mnrs', S_Pmn, S_PQinv, S_Pmn)

            return S4

        def compute_hartree(self, Cocc_a, Cocc_b=None):
            """
            Computes Hartree Operator in AO basis

            Parameters
            ----------
            ca: np.ndarray
                Occupied Orbitals in AO basis
            cb: np.ndarray
                if ref == 2, cb -> Beta Occupied Orbitals in AO basis
            """
            da = (Cocc_a @ Cocc_a.T)
            if Cocc_a is not None:
                db = (Cocc_b @ Cocc_b.T)
            else:
                db = da
            mf = dft.uks.UKS(self.mol)
            J = mf.get_j(dm=[da, db])

            return J
        
        def run_single_point(self, mol, basis, method):
            """
            Run a Standard calculation
            """

            # DFT
            molecule = gto.Mole() 
            molecule.atom = mol.atom
            molecule.basis = basis
            molecule.build()        

            if self.ref == 1:
                mf    = dft.RKS(mol)
            else:
                mf    = dft.UKS(mol)
            mf.xc = method
            mf.kernel()

            return mf.make_rdm1(), mf.mo_coeff, mf.mo_energy 


            # Post-SCF
            
        def diagonalize( self, matrix, ndocc ):
            """
            Diagonalizes Fock Matrix

            Parameters
            ----------
            marrix: np.ndarray
                Matrix to be diagonalized
            ndocc: int
                Number of occupied orbitals

            Returns
            -------
            C: np.ndarray
                Orbital Matrix
            Cocc: np.ndarray
                Occupied Orbital Matrix
            D: np.ndarray
                Density Matrix
            eigves: np.ndarray
                Eigenvalues
            """

            self.A = self.get_A()

            Fp = self.A.dot(matrix).dot(self.A)
            eigvecs, Cp = np.linalg.eigh(Fp)
            C = self.A.dot(Cp)
            Cocc = C[:, :ndocc]
            D = contract('pi,qi->pq', Cocc, Cocc)
            return C, Cocc, D, eigvecs
        