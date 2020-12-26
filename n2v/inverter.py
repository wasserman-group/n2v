"""
inverter.py
Density-to-potential inversion module

Handles the primary functions
"""

import numpy as np
from scipy.optimize import minimize
from opt_einsum import contract

import psi4
psi4.core.be_quiet()


def Inverter()

    def __init__(self, mol, basis_str, nt, aux="same"):

        self.basis_str = basis_str
        self.aux_str   = aux_str

        self.mol       = mol
        self.nt        = nt

        #Verify reference
        self.ref = psi4.core.get_global_option("REFERENCE")
        if self.ref == "UKS" or self.ref == "UHF" and len(nt) == 1:
            raise ValueError("Reference is set as Unrestricted but target density's dimension == 1")
        if self.ref == "RKS" or self.ref == "RHF" and len(nt) == 2:
            raise ValueError("Reference is set as Restricted but target density's dimension == 2")

        #Generate Basis set
        self.build_basis()


    def build_basis(self):
        """
        Build basis set object and auxiliary basis set object
        """

        basis = psi4.core.BasisSet.build( mol, key='BASIS', target=self)
        self.basis = basis
        self.nbf   = self.basis.nbf()
        self.naux  = self.basis.nbf()

        if self.aux_str is not "same":
            aux_basis = psi4.core.BasisSet.build( self.mol, key='Basis', target=self.aux_str)
            self.aux = aux_basis
            self.naux = aux_basis.nbf()






def canvas(with_attribution=True):
    """
    Placeholder function to show example docstring (NumPy format)

    Replace this function and doc string for your own project

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    quote = "The code is but a canvas to our imagination."
    if with_attribution:
        quote += "\n\t- Adapted from Henry David Thoreau"
    return quote


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print(canvas())
