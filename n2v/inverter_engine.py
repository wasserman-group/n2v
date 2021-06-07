"""
Inverter.py
"""
from warnings import warn
from dataclasses import dataclass
import numpy as np
from opt_einsum import contract

from .engines import Psi4Engine
from .methods.zmp import ZMP

@dataclass
class V:
    """Stores Potentials on AO"""
    T : np.ndarray
    pass

class E:
    """Stores Energies"""
    pass

class InverterEngine(ZMP):

    def __init__( self, engine='psi4' ):
        if engine == 'psi4':
            self.eng = Psi4Engine()

    def __repr__( self ):
        return "n2v.Inverter"

    def set_system( self, molecule, basis, ref=1, pbs='same' ):
        # Communicate TO engine
        self.mol_str = molecule.save_string_xyz()
        self.eng.mol  = molecule
        self.eng.basis_str = basis
        self.eng.pbs_str   = basis if pbs == 'same' else pbs
        self.ref = ref
        self.eng.ref = ref

        # self.nalpha = None
        # self.nbeta = None

        # Initialize ecompasses everything the engine builds with basis set 
        self.eng.initialize()
        self.set_basis_matrices()

        # Receive FROM engine
        self.nbf  = self.eng.nbf
        self.npbs = self.eng.npbs

    def set_basis_matrices( self ):
        """Generate basis dependant matrices"""
        self.T  = self.eng.get_T()
        self.V  = self.eng.get_V()
        self.A  = self.eng.get_A()
        self.S2 = self.eng.get_S()

        if self.eng.basis_str != 'same':  
            self.S3      = self.eng.get_S3()
            self.T_pbas  = self.eng.get_Tpbas()

        self.S4 = None

    def compute_hartree( self, Cooc_a, Cooc_b ):
        return self.eng.compute_hartree(Cooc_a, Cooc_b)

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

        Fp = self.A.dot(matrix).dot(self.A)
        eigvecs, Cp = np.linalg.eigh(Fp)
        C = self.A.dot(Cp)
        Cocc = C[:, :ndocc]
        D = contract('pi,qi->pq', Cocc, Cocc)
        return C, Cocc, D, eigvecs
    
    # Actual Methods
    def generate_components(self, guide_components, **keywords):
        """ I generate """
        self.guide_components = guide_components
        self.va = np.zeros( (self.nbf, self.nbf) )
        self.vb = np.zeros( (self.nbf, self.nbf) )
        self.J0 = self.compute_hartree(self.ct[0], self.ct[1])
        N       = self.nalpha + self.nbeta

        if guide_components == 'none':
            warn("Convergence will likely not be achieved")
        elif guide_components == 'hartree':
            self.va += self.J0[0]
            self.vb += self.J0[1]
        elif guide_components == 'fermi_amaldi':
            "Im calculating Fermi Amaldi Component"
            print("Computing Fermi Amaldi")
            v_fa = (1-1/N) * (self.J0[0] + self.J0[1])
            self.va += v_fa
            self.vb += v_fa
        else:
            raise ValueError("Guide component not recognized")

    def invert(self, method='zmp', 
                     guide_components='hartree',
                     **keywords):
        """I invert"""

        self.generate_components(guide_components)

        if method.lower() == 'zmp':
            self.zmp(**keywords)


