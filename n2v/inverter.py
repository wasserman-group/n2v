"""
Inverter.py
"""
from warnings import warn
from dataclasses import dataclass
import numpy as np
from opt_einsum import contract

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

class E:
    """Stores Energies"""

class Inverter(Direct, ZMP, WuYang, PDECO, OC, MRKS):
    """
    Attributes:
    ----------

    mol : Engine.molecule
        Molecule class of engine used
    basis : Engine.basis
        Basis class of engine used
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
    Dt : List
        List of np.ndarray for target density matrices (on AO).
    ct : List
        List of np.ndarray for input occupied orbitals. This might not be correct for post-HartreeFock methods.
    pbs_str: string
        name of Potential basis set
    pbs : Engine.basis
        Basis class for Potential basis set of the engine used. 
    npbs : int
        the length of pbs
    v_pbs : np.ndarray shape (npbs, ) for ref==1 and (2*npbs, ) for ref==2.
        potential vector on the Potential Baiss Set.
        If the potential is not represented on the basis set, this should
        remain 0. It will be initialized to a 0 array. One can set this
        value for initial guesses before Wu-Yang method (WY) or PDE-Constrained
        Optimization method (PDE-CO). For example, if PDE-CO is ran after
        a WY calculation, the initial for PDE-CO will be the result of WY
        if v_pbs is not zeroed.
    S2  : np.ndarray
        The ao overlap matrix (i.e. S matrix)
    S3  : np.ndarray
        The three ao overlap matrix (ao, ao, pbs)
    S4  : np.ndarray
        The four ao overlap matrix, the size should be (ao, ao, ao, ao)
    jk  : Engine.jk
        Engine jk object.
    T   : np.ndarray
        kinetic matrix on ao
    V   : np.ndarray
        external potential matrix on ao
    T_pbs: np.ndarray
        kinetic matrix on pbs. Useful for regularization.
    guide_potential_components: list of string
        guide potential components name
    va, vb: np.ndarray of shape (nbasis, nbasis)
        guide potential Fock matrix.
    """

    def __init__( self, engine='psi4' ):
        self.eng_str = engine.lower()
        if engine.lower() == 'psi4':
            from .engines import Psi4Engine
            self.eng = Psi4Engine()
        elif engine.lower() == 'pyscf':
            from .engines import PySCFEngine
            self.eng = PySCFEngine()
        else:
            raise ValueError("Engine name is incorrect. The availiable engines are: {psi4, pyscf}")
            
    def __repr__( self ):
        return "n2v.Inverter"

    def set_system( self, molecule, basis, ref=1, pbs='same' , **kwargs):
        """
        Stores relevant information and intitializes Engine

        Parameters
        ----------
        molecule: Engine.molecule
            Molecule object of selected engine
        basis: str
            Basis set of the main calculation
        ref: int
            reference for system. Restricted   -> 1
                                  Unrestricted -> 2
        pbs: str, default='same'
            Basis set for the potential
        **kwargs:
            Optional Parameters for different Engiens 
            Psi4 Engine:
                wfn : psi4.core.{RHF, UHF, RKS, UKS, Wavefunction, CCWavefuncion...}
                Psi4 wavefunction object
            PySCF Engine:
                None

        """
        # Communicate TO engine

        self.eng.set_system(molecule, basis, ref, pbs, **kwargs)
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
        from .engines import Psi4Engine
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
        """
        Generate basis dependant matrices
        """
        self.T  = self.eng.get_T()
        self.V  = self.eng.get_V()
        self.A  = self.eng.get_A()
        self.S2 = self.eng.get_S()
        self.S3      = self.eng.get_S3()

        if self.eng.pbs_str != 'same':  
            self.T_pbs  = self.eng.get_Tpbas()

        self.S4 = None

    def compute_hartree( self, Cocc_a, Cocc_b ):
        """
        Computes Hartree Potential on AO basis set. 

        Parameters
        ----------
        Cocc_a, Cocc_b: np.ndarray (nbf, nbf)
            Occupied orbitals in ao basis

        Returns
        -------
        J: List of np.ndarray
            Hartree potential due to density from Cocc_a and Cocc_b
        """
        return self.eng.compute_hartree(Cocc_a, Cocc_b )

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
        Stores values in object. 

        Parameters
        ----------
        v: np.ndarray
            Additional external potential to be added to hamiltonian along with:
            Kinetic_nm
            External_nm
            Guide_Potential_nm
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
        """
        Generates exact potential components to be added to
        the Hamiltonian to aide in the inversion procedure. 
        Parameters:
        -----------
        guide_potential_components: list
            Components added as to guide inversion. 
            Can be chosen from ["hartree", "fermi_amandi", "svwn"]
        """

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

    def invert(self, method, 
                     guide_components = 'hartree',
                     opt_max_iter = 50,
                     **keywords):
        """
        Handler to all available inversion methods
        Parameters
        ----------
        method: str
            Method used to invert density. 
            Can be chosen from {wuyang, zmp, mrks, oc}. 
            See documentation below for each method. 
        guide_components: list, opt
            Components added as to guide inversion. 
            Can be chosen from {"fermi_amandi", "svwn"}
            Default: ["fermi_amaldi"]
        opt_max_iter: int, opt
            Maximum number of iterations inside the chosen inversion.
            Default: 50
        direct
        ------
        Direct inversion of a set of Kohn-Sham equations. 
        $$v_{xc}(r) = \frac{1}{n(r)} \sum_i^N [\phi_i^{*} (r) \nabla^2 \phi_i(r) + \varepsilon_i | \phi_i(r)|^2] $$
            Parameters:
            -----------
                grid: np.ndarray, opt
                    Grid where result will be expressed in.
                    If not provided, dft grid will be used instead. 
                
        wuyang
        ------
        the Wu-Yang method:
        The Journal of chemical physics 118.6 (2003): 2498-2509.
            Parameters:
            ----------
                opt_max_iter: int
                    maximum iteration
                opt_method: string, opt
                    Method for scipy optimizer
                    Currently only used by wuyang and pdeco method.
                    Defaul: 'trust-krylov'
                    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
                reg : float, opt
                    Regularization constant for Wuyant Inversion.
                    Default: None -> No regularization is added.
                    Becomes attribute of inverter -> inverter.lambda_reg
                tol: float
                    tol for scipy.optimize.minimize
                gtol: float
                    gtol for scipy.optimize.minimize: the gradient norm for
                    convergence
                opt: dict
                    options for scipy.optimize.minimize
                                Notice that opt has lower priorities than opt_max_iter and gtol.
            return:
                the result are stored in self.v_pbs
            
        zmp
        ---
        The Zhao-Morrison-Parr Method:
        Phys. Rev. A 50, 2138 
            Parameters:
            ----------
                lambda_list: list
                    List of Lamda parameters used as a coefficient for Hartree 
                    difference in SCF cycle.
                zmp_mixing: float, optional
                    mixing \in [0,1]. How much of the new potential is added in.
                    For example, zmp_mixing = 0 means the traditional ZMP, i.e. all the potentials from previous
                    smaller lambda are ignored.
                    Zmp_mixing = 1 means that all the potentials of previous lambdas are accumulated, the larger lambda
                    potential are meant to fix the wrong/inaccurate region of the potential of the sum of the previous
                    potentials instead of providing an entire new potentials.
                    default: 1
                opt_max_iter: float
                    Maximum number of iterations for scf cycle
                opt_tol: float
                    Convergence criteria set for Density Difference and DIIS error. 
                return:
                    The result will be stored in self.proto_density_a and self.proto_density_b
                    For zmp_mixing==1, restricted (ref==1):
                        self.proto_density_a = \sum_i lambda_i * (Da_i - Dt[0]) - 1/N * (Dt[0])
                        self.proto_density_b = \sum_i lambda_i * (Db_i - Dt[1]) - 1/N * (Dt[1]);
                    unrestricted (ref==1):
                        self.proto_density_a = \sum_i lambda_i * (Da_i - Dt[0]) - 1/N * (Dt[0] + Dt[1])
                        self.proto_density_b = \sum_i lambda_i * (Db_i - Dt[1]) - 1/N * (Dt[0] + Dt[1]);
                    For restricted (ref==1):
                        vxc = \int dr' \frac{self.proto_density_a + self.proto_density_b}{|r-r'|}
                            = 2 * \int dr' \frac{self.proto_density_a}{|r-r'|};
                    for unrestricted (ref==2):
                        vxc_up = \int dr' \frac{self.proto_density_a}{|r-r'|}
                        vxc_down = \int dr' \frac{self.proto_density_b}{|r-r'|}.
                    To get potential on grid, one needs to do
                        vxc = self.on_grid_esp(Da=self.proto_density_a, Db=self.proto_density_b, grid=grid) for restricted;
                        vxc_up = self.on_grid_esp(Da=self.proto_density_a, Db=np.zeros_like(self.proto_density_a),
                                  grid=grid) for unrestricted;
        mRKS
        ----
        the modified Ryabinkin-Kohut-Staroverov method:
        Phys. Rev. Lett. 115, 083001 
        J. Chem. Phys. 146, 084103p
            Parameters:
            -----------
                maxiter: int
                    same as opt_max_iter
                vxc_grid: np.ndarray of shape (3, num_grid_points), opt
                    When this is given, the final result will be represented
                v_tol: float, opt
                    convergence criteria for vxc Fock matrices.
                    default: 1e-4
                D_tol: float, opt
                    convergence criteria for density matrices.
                    default: 1e-7
                eig_tol: float, opt
                    convergence criteria for occupied eigenvalue spectrum.
                    default: 1e-4
                frac_old: float, opt
                    Linear mixing parameter for current vxc and old vxc.
                    If 0, no old vxc is mixed in.
                    Should be in [0,1)
                    default: 0.5.
                init: string or psi4.core.Wavefunction, opt
                    Initial guess method.
                    default: "SCAN"
                    1) If None, input wfn info will be used as initial guess.
                    2) If "continue" is given, then it will not initialize
                    but use the densities and orbitals stored. Meaningly,
                    one can run a quick WY calculation as the initial
                    guess. This can also be used to user speficified
                    initial guess by setting Da, Coca, eigvec_a.
                    3) If it's not continue, it would be expecting a
                    method name string that works for psi4. A separate psi4 calculation
                    would be performed.
                sing: tuple of float of length 4, opt.
                    Singularity parameter for _vxc_hole_quadrature()
                    default: (1e-5, 1e-4, 1e-5, 1e-4)
                    [0]: atol, [1]: atol1 for dft_spherical grid calculation.
                    [2]: atol, [3]: atol1 for vxc_grid calculation.
            return:
                The result will be stored in self.grid.vxc
        oc
        --
        Ou-Carter method
        J. Chem. Theory Comput. 2018, 14, 5680−5689
            Parameters:
            -----------
                maxiter: int
                    same as opt_max_iter
                vxc_grid: np.ndarray of shape (3, num_grid_points)
                    The final result will be represented on this grid
                    default: 1e-4
                D_tol: float, opt
                    convergence criteria for density matrices.
                    default: 1e-7
                eig_tol: float, opt
                    convergence criteria for occupied eigenvalue spectrum.
                    default: 1e-4
                frac_old: float, opt
                    Linear mixing parameter for current vxc and old vxc.
                    If 0, no old vxc is mixed in.
                    Should be in [0,1)
                    default: 0.5.
                init: string, opt
                    Initial guess method.
                    default: "SCAN"
                    1) If None, input wfn info will be used as initial guess.
                    2) If "continue" is given, then it will not initialize
                    but use the densities and orbitals stored. Meaningly,
                    one can run a quick WY calculation as the initial
                    guess. This can also be used to user speficified
                    initial guess by setting Da, Coca, eigvec_a.
                    3) If it's not continue, it would be expecting a
                    method name string that works for psi4. A separate psi4 calculation
                    would be performed.
                    wuyang
        pdeco
        -----
        the PDE-Constrained Optimization method:
        Int J Quantum Chem. 2018;118:e25425;
        Nat Commun 10, 4497 (2019).
            Parameters:
            ----------
                opt_max_iter: int
                    maximum iteration
                opt_method: string, opt
                    Method for scipy optimizer
                    Currently only used by wuyang and pdeco method.
                    Defaul: 'L-BFGS-B'
                    Options: ['L-BFGS-B', 'BFGS']
                    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
                reg : float, opt
                    Regularization constant for Wuyant Inversion.
                    Default: None -> No regularization is added.
                    Becomes attribute of inverter -> inverter.lambda_reg
                gtol: float
                    gtol for scipy.optimize.minimize: the gradient norm for
                    convergence
                opt: dict
                    options for scipy.optimize.minimize
                    Notice that opt has lower priorities than opt_max_iter and gtol.
            return:
                the result are stored in self.v_pbs
        """

        self.generate_components(guide_components)

        if method.lower() == "direct":
            return self.direct_inversion(**keywords)
        elif method.lower() == "wuyang":
            self.wuyang(opt_max_iter, **keywords)
        elif method.lower() == "zmp":
            self.zmp(opt_max_iter, **keywords)
        elif method.lower() == "mrks":
            if self.eng_str == 'pyscf':
                raise ValueError("mRKS method not yet available with the PySCF engine. Try another method or another engine.")
            return self.mRKS(opt_max_iter, **keywords)
        elif method.lower() == 'oc':
            if self.eng_str == 'pyscf':
                raise ValueError("OuCarter method not yet available with the PySCF engine. Try another method or another engine.")
            return self.oucarter(opt_max_iter, **keywords)
        elif method.lower() == 'pdeco':
            return self.pdeco(opt_max_iter, **keywords)
        else:
            raise ValueError(f"Inversion method not available. Methods available: {['wuyang', 'zmp', 'mrks', 'oc', 'pdeco']}")

