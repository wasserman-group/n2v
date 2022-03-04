"""
Provides interface n2v interface to Psi4
"""


from .engine import Engine
import numpy as np
from opt_einsum import contract

try:
    import psi4
    psi4.set_options({"save_jk" : True})
    has_psi4 = True
except ImportError:
    has_psi4 = False
    
if has_psi4:
    from ..grid import Psi4Grider
    class Psi4Engine(Engine):
        """
        Psi4 Engine Class
        """

        def set_system(self, molecule, basis, ref='1', pbs='same', wfn=None):
            """
            Initializes geometry and basis infromation

            Parameters
            ----------
            molecule: psi4.core.Molecule
                Molecule of the system  used
            basis: str
                Basis set of calculation
            ref: int
                Reference: Restricted   -> 1
                           Unrestricted -> 2
            pbs: str
                Basis set of potential used
            wfn : psi4.core.{RHF, UHF, RKS, UKS, Wavefunction, CCWavefuncion...}
                Psi4 wavefunction object
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

            self.nalpha = wfn.nalpha()
            self.nbeta = wfn.nbeta()

            self.wfn = wfn

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
            """Overlap matrix in AO basis"""
            return np.array( self.mints.ao_overlap() )

        def get_S3(self):
            """3 Orbitals Overlap matrix in AO basis"""
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

        def hartree_NO(self, Dta):
            """
            Computes Hartree potential in AO basis from Natural Orbitals
            """

            if self.wfn is None:
                raise ValueError('Please provide a wfn object to the Inverter, i.e., Inverter.eng = wfn')

            if type(self.wfn) == psi4.core.CCWavefunction:
                C_NO    = psi4.core.Matrix(self.nbf, self.nbf)
                eigs_NO = psi4.core.Vector(self.nbf)
                self.wfn.Da().diagonalize( C_NO, eigs_NO, psi4.core.DiagonalizeOrder.Descending )
                occ = np.sqrt( np.array(eigs_NO) )
                new_CA = occ * np.array(C_NO)
                assert np.allclose( new_CA @ new_CA.T, Dta )
                if self.ref == 1:
                    new_CB = np.copy( new_CA )
                else:
                    self.wfn.Db().diagonalize( C_NO, eigs_NO, psi4.core.DiagonalizeOrder.Descending )
                    occ_b = np.sqrt( np.array( eigs_NO ) )
                    new_CB = occ_b * np.array( C_NO )
                J0 = self.compute_hartree(new_CA, new_CB)
                return J0

        def run_single_point(self, mol, basis, method):
            """
            Run a standard energy calculation
            """

            wfn_temp = psi4.energy(init+"/" + self.basis_str, 
                                molecule=self.mol, 
                                return_wfn=True)[1]

            if self.ref == 1:
                D = np.array(wfn_temp.Da()) + np.array(wfn_temp.Db())
                C = np.array(wfn_temp.Ca())
                e = np.array(wfn_temp.epsilon_a())
                
            else:
                D = np.stack( (np.array(wfn_temp.Da()), np.array(wfn_temp.Db())) )
                C = np.stack( (np.array(wfn_temp.Ca()), np.array(wfn_temp.Cb())) )
                e = np.stack( (np.array(wfn_temp.epsilon_a()), np.array(wfn_temp.epsilon_b())) )

            return D, C, e
            