"""
inverter.py
Density-to-potential inversion module

Handles the primary functions
"""

import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize
from opt_einsum import contract

import sys

import psi4
psi4.core.be_quiet()
psi4.core.clean()

from .methods.wuyang import WuYang
from .methods.zmp import ZMP
from .methods.rmks import MRKS
from .grid.grider import Grider


@dataclass
class data_bucket:
    """
    Data class for storing grid attributes. 
    """
    pass


class Inverter(WuYang, ZMP, MRKS, Grider):
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
    nt : List
        List of psi4.core.Matrix for target densities
    ct : List
        List of psi4.core.Matrix for occupied orbitals
    pbs_str: string
        name of pbs
    pbs : psi4.core.BasisSet
        Potential basis set.
    v0  : np.ndarray
        Initial zero guess for optimizer
    S2  : np.ndarray
        The ao overlap matrix (i.e. S matrix)
    S3  : np.ndarray
        The three ao overlap matrix (ao, ao, pbs)
    jk  : psi4.core.JK
        Psi4 jk object. Built if wfn has no jk, otherwise use wfn.jk

    T   : np.ndarray
        kinetic matrix on ao
    V   : np.ndarray
        external potential matrix on ao
    T_pbs: np.ndarray
        kinetic matrix on pbs. Useful for regularization.

    va, vb: np.ndarray of shape (nbasis, nbasis)
        guide potential component of Fock matrix.
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
        self.nt        = [np.array(wfn.Da_subset("AO")), np.array(wfn.Db_subset("AO"))]
        self.ct        = [wfn.Ca_subset("AO", "OCC"), wfn.Cb_subset("AO", "OCC")]
        self.pbs       = self.basis if pbs == "same" \
                                    else psi4.core.BasisSet.build( self.mol, key='BASIS', target=self.pbs_str)
        self.npbs      = self.pbs.nbf()
        self.v0        = np.zeros( (self.npbs) ) if self.ref == 1 \
                                                 else np.zeros( 2 * self.npbs )
        self.generate_mints_matrices()

        self.grid = data_bucket
        self.cubic_grid = data_bucket

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
        self.A = A
        self.S3 = np.squeeze(mints.ao_3coverlap(self.basis,self.basis,self.pbs))

        #Core Matrices
        self.T = np.array(mints.ao_kinetic()).copy()
        self.V = np.array(mints.ao_potential()).copy()
        self.T_pbs = np.array(mints.ao_kinetic(self.pbs, self.pbs)).copy()

    def generate_jk(self, gen_K=False, memory=2.50e9):
        """
        Creates jk object for generation of Coulomb and Exchange matrices
        1.0e9 B -> 1.0 GB
        """
        jk = psi4.core.JK.build(self.basis)
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
        
        self.jk.C_left_add(Cocc_a)
        self.jk.C_left_add(Cocc_b) 
        self.jk.compute()
        self.jk.C_clear()

        J = [np.array(self.jk.J()[0]), np.array(self.jk.J()[1])]
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

        A = np.array(self.A).copy()
        Fp = A.dot(matrix).dot(A)
        eigvecs, Cp = np.linalg.eigh(Fp)
        C = A.dot(Cp)
        Cocc = C[:, :ndocc]
        D = contract('pi,qi->pq', Cocc, Cocc)
        return C, Cocc, D, eigvecs

    #------------->  Inversion:

    def invert(self, method,
                     opt_method='trust-krylov', 
                     guide_potential_components = ["fermi_amaldi"], 
                     opt_max_iter = 50,
                     opt_tol      = 1e-7,
                     reg=None,
                     lam=50):
        """
        Handler to all available inversion methods

        Parameters
        ----------

        method: str
            Method used to invert density. 
            Can be chosen from {wuyang, zmp, mrks}
        opt_method: string, opt
            Method for scipy optimizer
            Currently only used by wuyang method. 
            Defaul: 'trust-krylov'
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        guide_potential_components: list, opt
            Components added as to guide inversion. 
            Can be chosen from {"fermi_amandi", "svwn"}
            Default: ["fermi_amaldi"]
        opt_max_iter: int, opt
            Maximum number of iterations inside the chosen inversion.
            Default: 50
        reg = float, opt
            Regularization constant for Wuyant Inversion. 
            Default: None -> No regularization is added. 
            Becomes attribute of inverter -> inverter.lambda_reg
        lam = int, opt
            Lamda parameter for ZMP method. 
            Default: 50. May become unstable if lam is too big. 
            Becomes attirube of inverter -> inverter.lambda
        """

        self.lam = lam
        self.lambda_reg = reg
        self.generate_components(guide_potential_components)

        if method.lower() == "wuyang":
            self.wuyang(opt_method, opt_max_iter, opt_tol)
        elif method.lower() == "zmp":
            self.zmp_with_scf(lam, opt_max_iter, opt_tol)
        elif method.lower() == "mrks":
            pass
        else:
            raise ValueError(f"Inversion method not available. Try: {['wuyang', 'zmp', 'mrks']}")

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
        self.J0, _ = self.form_jk(self.ct[0], self.ct[1])

        if "fermi_amaldi" in guide_potential_components:
            v_fa = (1-1/N) * (self.J0[0] + self.J0[1])

            self.va += v_fa
            self.vb += v_fa

        if "svwn" in guide_potential_components:
            if "svwn" in guide_potential_components:
                _, wfn_0 = psi4.energy( "svwn"+"/"+self.basis_str, molecule=self.mol , return_wfn = True)

            if self.ref == 1:
                ntarget = psi4.core.Matrix.from_array( [ self.nt[0] + self.nt[1] ] )
                wfn_0.V_potential().set_D( [ntarget] )
                vxc_a = psi4.core.Matrix( self.nbf, self.nbf )
                wfn_0.V_potential().compute_V([vxc_a])
                self.va += vxc_a.np
                self.vb += vxc_a.np
            elif self.ref == 2:
                na_target = psi4.core.Matrix.from_array( self.nt[0] )
                nb_target = psi4.core.Matrix.from_array( self.nt[1] )
                wfn_0.V_potential().set_D( [na_target, nb_target] )
                vxc_a = psi4.core.Matrix( self.nbf, self.nbf )
                vxc_b = psi4.core.Matrix( self.nbf, self.nbf )
                wfn_0.V_potential().compute_V([vxc_a, vxc_b])
                self.va += vxc_a.np
                self.vb += vxc_b.np

    def finalize_energy(self):
        """
        Calculates energy contributions
        """

        target_one        = self.wfn.to_file()['floatvar']['ONE-ELECTRON ENERGY']
        target_two        = self.wfn.to_file()['floatvar']['TWO-ELECTRON ENERGY']

        energy_kinetic    = contract('ij,ij', self.T, (self.Da + self.Db))
        energy_external   = contract('ij,ij', self.V, (self.Da + self.Db))
        energy_hartree_a  = 0.5 * contract('ij,ji', self.Hartree_a + self.Hartree_b, self.Da)
        energy_hartree_b  = 0.5 * contract('ij,ji', self.Hartree_a + self.Hartree_b, self.Db)

        print("WARNING: XC Energy is not yet properly calculated")
        energy_ks = 0.0
        energies = {"One-Electron Energy" : energy_kinetic + energy_external,
                    "Two-Electron Energy" : energy_hartree_a + energy_hartree_b,
                    # "XC"                  : energy_ks,
                    # "Total Energy"        : energy_kinetic   + energy_external  + \
                    #                         energy_hartree_a + energy_hartree_b + \
                    #                         energy_ks 
                    }

        target_energies =  {"One-Electron Energy" : target_one,
                            "Two-Electron Energy" : target_two,
                           }

        self.energies = energies
        self.target_energies = target_energies

        # print(f"Final Energies: {self.energies}")
        # print(f"Target Energies: {self.target_energies}")