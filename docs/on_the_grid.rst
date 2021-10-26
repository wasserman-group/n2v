**On the Grid**
===============

An additional feature of n2v is the *Grider* class. This class has a handfull of 
options to represent many quantities and operators on the grid. 

In order to use them. One must specify either 1) a grid in the appropriate format
or 2) a V_potential object from a psi4 calculation using a DFT method or built 
from scratch. 

One can have access to the following quantities on the grid: density, orbitals,
external potential, hartree potential, exchange correlation potentials (Local only), 
laplacian of molecular orbitals, gradient of molecular orbitals. As well 
as being able to represent any quantity on the grid into ao basis set. 

