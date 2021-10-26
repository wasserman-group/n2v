**First Steps**
===============

This page details how to get started with *n2v*. 

*n2v* allows you to perform density-to-potential inversions. 
By default, we interface our package with the package Psi4. It is a good idea
to become familiarized with it and it's python interface. 

The first need we need to make a calculation is to obtain a target density. Let 
us do this in Psi4 for the Neon atom. 

.. code-block:: python

  import n2v
  import psi4

  # Define psi4 geometry. Symmetry needs to be set to C1. 
  Ne = psi4.geometry("""
  0 1 
  Ne
  units bohr
  symmetry c1
  """)

  # n2v uses psi4's reference option to select restricted or unrestricted calculation.
  psi4.set_options({"reference" : "rhf"})
  
  # storing the JK object (coulomb/exchange matrices) will save additional time.
  psi4.set_options({"save_jk" : True})

  # - Perform a calculation for a target density. 
  # - For some methods psi4 won't generate a post scf density. Thus calculating
  #   a property is necessary for some post-scf methods. 
  # - Obtain a wavefunction object (wfn) as well. 
  e, wfn = psi4.properties("CCSD/cc-pvdz", return_wfn=True, properties=["dipole"], molecule=Ne)

All of the information required to invert the calculated density into its effective potential
is stored in the wfn. With it, we can create an inverter object and start using 
n2v.

.. code-block:: python

  ine = n2v.Inverter(wfn, pbs='cc-pvdz') 
  """
  Often times, we expand the effective potential in a different, larger basis
  set, when this happens, we specify the name of the basis as an argument. 
  If the same basis is to be used, you can ommit the argument and the same basis 
  as the calculation will be used. 
  """