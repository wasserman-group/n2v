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
from .methods.mrks import MRKS
from .methods.direct import Direct

@dataclass
class V:
    """Stores Potentials on AO"""
    T : np.ndarray
    pass

class E:
    """Stores Energies"""
    pass

class Inverter(Direct, ZMP, WuYang, PDECO, OC, MRKS):

    def __init__( self, engine='psi4' ):
        self.eng_str = engine.lower()
        if engine.lower() == 'psi4':
            self.eng = Psi4Engine()
        elif engine.lower() == 'pyscf':
            self.eng = PySCFEngine()
        else:
            raise ValueError("Engine name is incorrect. The availiable engines are: {psi4, pyscf}")
            
    def __repr__( self ):
        return "n2v.Inverter"

    def set_system( self, molecule, basis, ref=1, pbs='same' , **kwargs):
        # Communicate TO engine

        self.eng.set_system(molecule, basis, ref, pbs, **kwargs)
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

    @classmethod
    def from_wfn( self, wfn, pbs='same' ):
        """
        Generates Inverter directly from wavefunction. 
        
        Parameters
        ----------
        wfn: Psi4.Core.{RHF, RKS, ROHF, CCWavefunction, UHF, UKS, CUHF}
            Wavefunction Object
        Returns
        -------
        inv: n2v.Inverter
            Inverter Object. 
        """

        inv = self( engine='psi4' )
        inv.eng = Psi4Engine()
        ref = 1 if wfn.to_file()['boolean']['same_a_b_dens'] else 2
        inv.set_system( wfn.molecule(), wfn.basisset().name(), pbs=pbs, ref=ref, wfn=wfn )
        inv.Dt = [ np.array(wfn.Da()), np.array(wfn.Db()) ]
        inv.ct = [ np.array(wfn.Ca_subset("AO", "OCC")), np.array(wfn.Cb_subset("AO", "OCC")) ]
        inv.et = [ np.array(wfn.epsilon_a_subset("AO", "OCC")), np.array(wfn.epsilon_b_subset("AO", "OCC")) ]
        inv.eng_str = 'psi4'
        inv.eng.wfn = wfn

        return inv
        
    def set_basis_matrices( self ):
        """Generate basis dependant matrices"""
        self.T  = self.eng.get_T()
        self.V  = self.eng.get_V()
        self.A  = self.eng.get_A()
        self.S2 = self.eng.get_S()
        self.S3      = self.eng.get_S3()

        if self.eng.pbs_str != 'same':  
            self.T_pbs  = self.eng.get_Tpbas()

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

    def diagonalize_with_potential_vFock(self, v=None):
        """
        Diagonalize Fock matrix with additional external potential
        """

        if v is None:
            fock_a = self.V + self.T + self.va
        else:
            if self.ref == 1:
                fock_a = self.V + self.T + self.va + v
            else:
                valpha, vbeta = v
                fock_a = self.V + self.T + self.va + valpha
                fock_b = self.V + self.T + self.vb + vbeta


        self.Ca, self.Coca, self.Da, self.eigvecs_a = self.diagonalize( fock_a, self.nalpha )

        if self.ref == 1:
            self.Cb, self.Cocb, self.Db, self.eigvecs_b = self.Ca.copy(), self.Coca.copy(), self.Da.copy(), self.eigvecs_a.copy()
        else:
            self.Cb, self.Cocb, self.Db, self.eigvecs_b = self.diagonalize( fock_b, self.nbeta )    

    # Actual Methods
    def generate_components(self, guide_components, **keywords):
        self.guide_components = guide_components
        self.va = np.zeros( (self.nbf, self.nbf) )
        self.vb = np.zeros( (self.nbf, self.nbf) )
        self.J0 = self.compute_hartree(self.ct[0], self.ct[1])
        N       = self.nalpha + self.nbeta

        if self.eng_str == 'psi4':
            J0_NO = self.eng.hartree_NO(self.Dt[0])
            self.J0 = J0_NO if J0_NO is not None else self.J0

        if guide_components == 'none':
            warn("No guide potential was provided. Convergence may not be achieved")
        elif guide_components == 'hartree':
            self.va += self.J0[0] + self.J0[1]
            self.vb += self.J0[0] + self.J0[1]
        elif guide_components == 'fermi_amaldi':
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

