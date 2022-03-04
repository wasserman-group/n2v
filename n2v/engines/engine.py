"""
engines.py
"""

from abc import ABC, abstractmethod

class Engine(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def set_system(self, molecule, basis, ref, pbs):
        """
        Set system into engine. It is expected in this function to format each
        component in the appropriate way for the engine. 
        The inverter will require as well the following data:
        mol_str: str
            Molecule information in string form
        nbf: int
            Number of basis functions
        npbs: int
            Number of basis functions in basis used to express the inverted
            potential
        """
        pass

    @abstractmethod
    def initialize(self):
        """
        Initializes different components needed for inversions. This can include
        the generation of the overlap and electric repulsion matrices. As well
        as setting Coulomb/Exchange objects
        """
        pass

    @abstractmethod
    def get_V(self):
        """
        Generates nuclear external potential on the atomic orbital basis set. 
        
        Parameters:
        ----------
        
        Returns:
        --------
        V: np.ndarray
            External potential on basis. Size: (nbf, nbf)
        """
        return V

    @abstractmethod
    def get_T(self):
        """
        Generates Kinetic contribution on the atomic orbital basis set. 
        
        Parameters:
        ----------
        
        Returns:
        --------
        T: np.ndarray
            Kinetic matrix on basis. Size: (nbf, nbf)
        """
        return T

    @abstractmethod
    def get_Tpbas(self):
        """
        Generates Kinetic contribution on the atomic orbital basis set used
        to express the inverted potential
        
        Parameters:
        ----------
        
        Returns:
        --------
        Tpbas: np.ndarray
            Kinetic matrix on secondary basis. Size: (npbas, npbas)
        """
        return Tpbas 

    @abstractmethod
    def get_A(self):
        """
        Generates nuclear external potential on the atomic orbital basis set. 
        
        Parameters:
        ----------
        
        Returns:
        --------
        V: np.ndarray
            External potential on basis. Size: (nbf, nbf)
        """
        return A

    @abstractmethod
    def get_S(self):
        """
        Generates the overlap matrix of the atomic orbital basis set. 
        
        Parameters
        ----------
        
        Returns:
        --------
        S: np.ndarray
            Overlap matrix. Size: (nbf, nbf)
        """
        return S

    @abstractmethod
    def get_S3(self):
        """
        Generates the three overlap matrix of the atomic orbital basis set with
        the basis used to express the inverted potential. 
        
        Parameters:
        ----------
        
        Returns:
        --------
        S3: np.ndarray
            Three overlap matrix. Size: (nbf, nbf, npbas)
        """
        return S3

    @abstractmethod
    def get_S4(self):
        """
        Generates the four overlap matrix of 4 different orbital basis sets. 
        
        Parameters:
        ----------
        
        Returns:
        --------
        S4: np.ndarray
            Four overlap matrix
        """
        return S4

    @abstractmethod
    def compute_hartree(self, Cocc_a, Cocc_b):
        """
        Generates the Hartree potential on the atomic orbital basis set
        
        Parameters:
        -----------
        Cocc_a, Cocc,b: np.ndarray. Size: (nbf, nbf)
            Occupied molecular orbitals (alpha, beta)

        Returns:
        --------
        J: List of lenght two with np.ndarrays. 
            Hartree potential on ao basis (alpha, beta)
        """
        return J
    






        