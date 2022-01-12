"""
Inverter.py
"""
from warnings import warn
from dataclasses import dataclass
import numpy as np
from opt_einsum import contract

from .engines import Psi4Engine
from .engines import PySCFEngine

from .methods.zmp import ZMP
from .methods.wuyang import WuYang
from .methods.pdeco import PDECO
from .methods.oucarter import OC

@dataclass
class V:
    """Stores Potentials on AO"""
    T : np.ndarray
    pass

class E:
    """Stores Energies"""
    pass

class Inverter(ZMP, WuYang, PDECO, OC):

    def __init__( self, engine='psi4' ):
        self.eng_str = engine
        if engine.lower() == 'psi4':
            self.eng = Psi4Engine()
        elif engine.lower() == 'pyscf':
            self.eng = PySCFEngine()
        else:
            raise ValueError("Engine name is incorrect. The availiable engines are: {psi4, pyscf}")
            
    def __repr__( self ):
        return "n2v.Inverter"

    def set_system( self, molecule, basis, ref=1, pbs='same' ):
        # Communicate TO engine

        self.eng.set_system(molecule, basis, ref, pbs)
        # self.mol_str = self.eng.mol_str
        self.ref = ref

        self.nalpha = self.eng.nalpha
        self.nbeta = self.eng.nbeta

        # Initialize ecompasses everything the engine builds with basis set 
        self.eng.initialize()
        self.set_basis_matrices()

        # Receive FROM engine
        self.nbf  = self.eng.nbf
        self.npbs = self.eng.npbs
        self.v_pbs = np.zeros( (self.npbs) ) if self.ref == 1 \
                                             else np.zeros( 2 * self.npbs )

    def set_basis_matrices( self ):
        """Generate basis dependant matrices"""
        self.T  = self.eng.get_T()
        self.V  = self.eng.get_V()
        self.A  = self.eng.get_A()
        self.S2 = self.eng.get_S()
        self.S3      = self.eng.get_S3()

        if self.eng.pbs_str != 'same':  
            self.T_pbas  = self.eng.get_Tpbas()

        self.S4 = None

    def compute_hartree( self, *args ):
        return self.eng.compute_hartree(*args)

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
        self.guide_components = guide_components
        self.va = np.zeros( (self.nbf, self.nbf) )
        self.vb = np.zeros( (self.nbf, self.nbf) )
        self.J0 = self.compute_hartree(self.ct[0], self.ct[1])
        N       = self.nalpha + self.nbeta

        if guide_components == 'none':
            warn("No guide potential was provided. Convergence may not be achieved")
        elif guide_components == 'hartree':
            self.va += self.J0[0]
            self.vb += self.J0[1]
        elif guide_components == 'fermi_amaldi':
            print("Computing Fermi Amaldi")
            v_fa = (1-1/N) * (self.J0[0] + self.J0[1])
            self.va += v_fa
            self.vb += v_fa
        else:
            raise ValueError("Guide component not recognized")

    def invert(self, method = 'zmp', 
                     guide_components = 'hartree',
                     opt_max_iter = 50,
                     **keywords):
        """"""

        self.generate_components(guide_components)

        if method.lower() == "direct":
            return self.direct_inversion(**keywords)
        elif method.lower() == "wuyang":
            self.wuyang(opt_max_iter, **keywords)
        elif method.lower() == "zmp":
            self.zmp(opt_max_iter, **keywords)
        elif method.lower() == "mrks":
            return self.mRKS(opt_max_iter, **keywords)
        elif method.lower() == 'oc':
            return self.oucarter(opt_max_iter, **keywords)
        elif method.lower() == 'pdeco':
            return self.pdeco(opt_max_iter, **keywords)
        else:
            raise ValueError(f"Inversion method not available. Methods available: {['wuyang', 'zmp', 'mrks', 'oc', 'pdeco']}")

