Direct
======

The direct method uses the "Kohn-Sham Inversion Formula" and it is applicable
when the set of Kohn-Sham orbitals is available. By manipulating the Kohn–Sham
equations we find that the potential is expressed by:

.. math::
  v_{xc}(r) = \frac{1}{n(r)} \sum_i^N [\phi_i^{*} (r) \nabla^2 \phi_i(r) + \varepsilon_i | \phi_i(r)|^2]

The full method's description can be found:
A.  A.  Kananenka,  S.  V.  Kohut,  A.  P.  Gaiduk,  I.  G.  Ryabinkin,  and  V.  N.  Staroverov, 
“Efficient construction of exchange and correlation potentials by inverting the Kohn–Sham equations”,
The Journal of chemical physics, vol. 139, no. 7, p. 074 112, 2013.


In *n2v*, we would can request the method in the following way.
Let's assume we want the Kohn-Sham potential of a Neon Atom. 

.. code-block:: python

    #Define Psi4 geometries. Symmetries need to be set to C1!
    Ne = psi4.geometry( 
    """ 
    0 1
    Ne 0.0 0.0 0.0
    noreorient
    nocom
    units bohr
    symmetry c1
    """ )

    #Perform a DFT calculation.
    e, wfn = psi4.energy("pbe/6-311G", return_wfn=True, molecule=Ne)

    #Define inverter objects for each molcule. Simply use the wnf object from psi4 as an argument. 
    ine = n2v.Inverter(wfn)

    vxc_inverted = ine.invert('direct', grid=grid)


This method is known for introducing spurious oscillations that are due to 
defficiencies in the basis set used. In order to supress them, we can obtain an 
oscillatory profile and subtract it from the resulting potential. 

.. code-block:: python

  osc_profile = ine.get_basis_set_correction(grid)
  vxc_inverted -= osc_profile


This can be acomplished as well by simply using the argument ```correction```
in the function

.. code-block:: python

  vxc_inverted = ine.invert('direct', grid=grid, correction=True)

Notice that the resulting potential is produced in the specified grid and not
directly in the ao basis. 



