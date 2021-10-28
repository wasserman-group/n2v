Wu Yang
=======

The Wu and Yang method is a direct optimization that works by building a variational functional. 
The determination of the potential is turned into the maximization of this functional of the potential. 

The full method's description can be found:
Wu, Qin, and Weitao Yang. 
"A direct optimization method for calculating density functionals and exchange–correlation potentials from electron densities." 
The Journal of chemical physics 118.6 (2003): 2498-2509.

The method works by building a functional of the potential. 

.. math::
    W[\Psi_{det}[v_{KS}], v_{KS}] = T_s[\Psi_{det}] + \int d\mathbf{r} v_{KS}(\mathbf{r})\{n(\mathbf{r})-n_{in}(\mathbf{r})\}


At the minimum, this functional should be stationary withn respect to any variation of the potential. So that:

.. math::
    \frac{\delta W[\Psi_{det}[v_{KS}], v_{KS}]}{\delta v_{KS}(\mathbf{r})} = n(\mathbf{r})-n_{in}(\mathbf{r})

Which is just the constraint condition of the electron density. Additionally we may require the second-order functional derivative. 
In practice, we express it in terms of the orbitals through first order perturbation theory:

.. math::
	\frac{\delta^2 W[\Psi_{det}[v_{KS}], v_{KS}]}{\delta v_{KS}(\mathbf{r})\delta v_{KS}(\mathbf{r'})} =  \frac{\delta n(\mathbf{r})}{\delta v_{KS}(\mathbf{r'})} = 2\sum_i^{occ.}\sum_a^{unocc.} \frac{\psi_i^*(\mathbf{r})\psi_a(\mathbf{r})\psi_i(\mathbf{r'})\psi_a^*(\mathbf{r'})}{\epsilon_i - \epsilon_a}.


With the functional, its gradient and hessian. We can use an optimizer to optimize the functional. In *n2v* we use the different optimizers from *Scipy*. 
Here is how it works. 

.. code-block:: python 

    Be = psi4.geometry( 
    """ 
    0 1
    Be
    noreorient
    nocom
    units bohr
    symmetry c1
    """ )

    psi4.set_options({"reference" : "rhf"})  # Spin-Restricted

    # IMPORTANT NOTE: psi4.energy does not update cc densities. So we calculate dipole moments instead.
    wfn = psi4.properties("ccsd/aug-cc-pvtz",  return_wfn=True, molecule=Be, property=['dipole'])[1]

    # Build inverter and set target
    ibe = n2v.Inverter(wfn, pbs="aug-cc-pvqz")

And just like we do in the ZMP method, we can add to the potential all the components that we know exactly. 
So that the Kohn-Sham potential, if we are using the Fermi Amaldi potential as the guide, can be express as: 

.. math::
  v_{Kohn-Sham}=v_{ext}+v_{guide}+v_{rest}

.. code-block:: python

  # Inverter with WuYang method, guide potention v0=Fermi-Amaldi
  ibe.invert("WuYang", guide_potential_components=["fermi_amaldi"])

The resulting potential will be found on ```inverter.v_pbs``` which can be turned back into its repressentation in space to be visualized:

.. code-block:: python

   vrest = ibe.on_grid_ao(ibe.v_pbs, grid=grids, basis=ibe.pbs)  # Note that specify the basis set 
                                                                 # that vrest is on.

The full example can be found in the *n2v examples repository*.
