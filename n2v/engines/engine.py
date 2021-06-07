"""
engines.py
"""

from abc import ABC, abstractmethod

class Engine(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def get_V(self):
        return V

    @abstractmethod
    def get_T(self):
        return T

    @abstractmethod
    def get_Tpbas(self):
        return Tpbas 

    @abstractmethod
    def get_A(self):
        return A

    @abstractmethod
    def get_S(self):
        return S

    @abstractmethod
    def get_S3(self):
        return S3

    @abstractmethod
    def get_S4(self):
        return S4

    @abstractmethod
    def compute_hartree(self, Cocc_a, Cocc_b):
        return J






        