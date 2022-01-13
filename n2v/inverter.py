"""
inverter.py
Density-to-potential inversion module

Handles the primary functions
"""

import numpy as np
from dataclasses import dataclass
from opt_einsum import contract

import psi4
psi4.core.be_quiet()
psi4.core.clean()
psi4.set_options({"save_jk" : True})

from .methods.wuyang import WuYang
from .methods.zmp import ZMP
from .methods.mrks import MRKS
from .methods.oucarter import OC
from .methods.pdeco import PDECO
from .methods.direct import Direct
from .grid.grider import Grider

@dataclass
class data_bucket:
    """
    Data class for storing grid attributes. 
    """
    pass


class Inverter(Direct, WuYang, ZMP, MRKS, OC, PDECO, Grider):
    """
    Attributes:
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
    Dt : List
        List of psi4.core.Matrix for target density matrices (on AO).
    ct : List
        List of psi4.core.Matrix for input occupied orbitals. This might not be correct for post-HartreeFock methods.
    pbs_str: string
        name of Potential basis set
    pbs : psi4.core.BasisSet
        Potential basis set.
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
    jk  : psi4.core.JK
        Psi4 jk object. Built if wfn has no jk, otherwise use wfn.jk

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
    Methods:
    --------

    """
    def __init__(self, wfn, pbs="same", debug=False):
        """
        Handles Inversion
        
        Parameters
        ----------
        wfn : psi4.core.{RHF, UHF, RKS, UKS, Wavefunction, CCWavefuncion...}
            Psi4 wavefunction object
        pbs: str. default="same". If same, then the potential basis set (pbs)
                 is the same as orbital basis set (i.e. ao). Notice that
                 pbs is not needed for some methods
        """
        self.wfn       = wfn
        self.pbs_str   = pbs
        self.mol       = wfn.molecule()
        self.basis     = wfn.basisset()
        self.basis_str = wfn.basisset().name()
        self.nbf       = wfn.basisset().nbf()
        self.nalpha    = wfn.nalpha()
        self.nbeta     = wfn.nbeta()
        self.ref       = 1 if psi4.core.get_global_option("REFERENCE") == "RHF" or \
                              psi4.core.get_global_option("REFERENCE") == "RKS" else 2
        self.jk        = wfn.jk() if hasattr(wfn, "jk") == True else self.generate_jk()
        self.Dt        = (np.array(wfn.Da_subset("AO")), np.array(wfn.Db_subset("AO")))
        self.ct        = (np.array(wfn.Ca_subset("AO", "OCC")), np.array(wfn.Cb_subset("AO", "OCC")))
        self.pbs       = self.basis if pbs == "same" \
                                    else psi4.core.BasisSet.build( self.mol, key='BASIS', target=self.pbs_str)
        self.npbs      = self.pbs.nbf()
        self.v_pbs     = np.zeros( (self.npbs) ) if self.ref == 1 \
                                                 else np.zeros( 2 * self.npbs )
        self.generate_mints_matrices()
        self.grid = data_bucket
        self.cubic_grid = data_bucket
        
        self.J0 = None
        self.S4 = None  # Entry to save the 4 overlap matrix.

    #------------->  Basics:

    def generate_mints_matrices(self):
        """
        Generates matrices that are methods of a mints object
        """

        mints = psi4.core.MintsHelper( self.basis )

        #Overlap Matrices
        self.S2 = np.array(mints.ao_overlap())
        A = mints.ao_overlap()
        A.power( -0.5, 1e-16 )
        self.A = np.array(A)
        self.S3 = np.squeeze(mints.ao_3coverlap(self.basis,self.basis,self.pbs))

        #Core Matrices
        self.T = np.array(mints.ao_kinetic()).copy()
        self.V = np.array(mints.ao_potential()).copy()
        self.T_pbs = np.array(mints.ao_kinetic(self.pbs, self.pbs)).copy()

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

    def form_jk(self, Cocc_a, Cocc_b):
        """
        Generates Coulomb and Exchange matrices from occupied orbitals
        """
        # if  self.jk.memory_estimate() > psi4.get_memory() * 0.8:
        #     raise ValueError("Requested JK will take too more memory than default. \n \
        #                       Increase it with psi4.set_memory(int( Many More Bytes )))!")
        Cocc_a = psi4.core.Matrix.from_array(Cocc_a)
        Cocc_b = psi4.core.Matrix.from_array(Cocc_b)
        self.jk.C_left_add(Cocc_a)
        self.jk.C_left_add(Cocc_b) 
        self.jk.compute()
        self.jk.C_clear()

        J = (np.array(self.jk.J()[0]), np.array(self.jk.J()[1]))
        K = []

        return J, K

    def diagonalize(self, matrix, ndocc):
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

        A = self.A
        Fp = A.dot(matrix).dot(A)
        eigvecs, Cp = np.linalg.eigh(Fp)
        C = A.dot(Cp)
        Cocc = C[:, :ndocc]
        D = contract('pi,qi->pq', Cocc, Cocc)
        return C, Cocc, D, eigvecs

    #------------->  Inversion:

    def invert(self, method,
                     guide_potential_components = ["fermi_amaldi"],
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
        guide_potential_components: list, opt
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

        #Generate Guide Potential
        if method.lower()=='mrks':
            if guide_potential_components[0] != 'hartree' or len(guide_potential_components) != 1:
                print("The guide potential is changed to v_hartree.")
            self.generate_components(["hartree"])
        elif method.lower() != 'direct':
            self.generate_components(guide_potential_components)

        #Invert
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

    def generate_components(self, guide_potential_components):
        """
        Generates exact potential components to be added to
        the Hamiltonian to aide in the inversion procedure. 

        Parameters:
        -----------
        guide_potential_components: list
            Components added as to guide inversion. 
            Can be chosen from {"fermi_amandi", "svwn"}
        """

        self.va = np.zeros_like(self.T)
        self.vb = np.zeros_like(self.T)

        N = self.nalpha + self.nbeta

        if self.J0 is None:
            if type(self.wfn) == psi4.core.CCWavefunction:
                nbf = self.nbf
                C_NO = psi4.core.Matrix(nbf, nbf)
                eigs_NO = psi4.core.Vector(nbf)
                self.wfn.Da().diagonalize(C_NO, eigs_NO, psi4.core.DiagonalizeOrder.Descending)
                occu = np.sqrt(eigs_NO.np)
                New_Orb_a = occu * C_NO.np
                assert np.allclose(New_Orb_a @ New_Orb_a.T, self.Dt[0])
                if self.ref == 1:
                    New_Orb_b = New_Orb_a
                else:
                    self.wfn.Db().diagonalize(C_NO, eigs_NO, psi4.core.DiagonalizeOrder.Descending)
                    occu = np.sqrt(eigs_NO.np)
                    New_Orb_b = occu * C_NO.np
                self.J0 = self.form_jk(New_Orb_a, New_Orb_b)[0]
            else:
                self.J0, _ = self.form_jk(self.ct[0], self.ct[1])

        if "fermi_amaldi" in guide_potential_components:
            v_fa = (1-1/N) * (self.J0[0] + self.J0[1])

            self.va += v_fa
            self.vb += v_fa
        elif "hartree" in guide_potential_components:
            v_hartree = (self.J0[0] + self.J0[1])

            self.va += v_hartree
            self.vb += v_hartree
        else:
            raise ValueError("Hartee nor FA was included."
                             "Convergence will likely not be achieved")

        if "svwn" in guide_potential_components:
            if "svwn" in guide_potential_components:
                _, wfn_0 = psi4.energy( "svwn"+"/"+self.basis_str, molecule=self.mol , return_wfn = True)

            if self.ref == 1:
                ntarget = psi4.core.Matrix.from_array( [ self.Dt[0] + self.Dt[1] ] )
                wfn_0.V_potential().set_D( [ntarget] )
                vxc_a = psi4.core.Matrix( self.nbf, self.nbf )
                wfn_0.V_potential().compute_V([vxc_a])
                self.va += vxc_a.np
                self.vb += vxc_a.np
            elif self.ref == 2:
                na_target = psi4.core.Matrix.from_array( self.Dt[0] )
                nb_target = psi4.core.Matrix.from_array( self.Dt[1] )
                wfn_0.V_potential().set_D( [na_target, nb_target] )
                vxc_a = psi4.core.Matrix( self.nbf, self.nbf )
                vxc_b = psi4.core.Matrix( self.nbf, self.nbf )
                wfn_0.V_potential().compute_V([vxc_a, vxc_b])
                self.va += vxc_a.np
                self.vb += vxc_b.np
        self.guide_potential_components = guide_potential_components
