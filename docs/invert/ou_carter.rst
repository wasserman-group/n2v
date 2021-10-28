Ou and Carter
=============

Ou and Carter developed an inversion method inspired by the mRKS.  

Ou, Qi, and Emily A. Carter.  
"Potential Functional Embedding Theory with an Improved Kohn–Sham Inversion Algorithm."  
Journal of chemical theory and computation 14.11 (2018): 5680-5689.


In this method we are only required to have a KS set of equations. Then we can derive a self-consisent calculation for the exchange correlation potential as:

.. math::
  v_{xc}(\mathbf{r}) = \bar{\epsilon}^{KS}(\mathbf{r}) - \frac{\tau^{KS}_L(\mathbf{r})}{n^{KS}(\mathbf{r})} -v_{ext}(\mathbf{r}) - v_{H}(\mathbf{r})

where

.. math::
  \frac{\tau^{KS}_L(\mathbf{r})}{n^{KS}(\mathbf{r})} = 
  \frac{|\nabla n^{KS}(\mathbf{r})|^2}{8|n^{KS}(\mathbf{r})|^2}
  -\frac{\nabla^2 n^{KS}(\mathbf{r})}{4n^{KS}(\mathbf{r})}
  +\frac{\tau^{KS}_P(\mathbf{r})}{n^{KS}(\mathbf{r})}.


By replacing the Kohn-Sham density everywhere by the accurate input density and the external potential by an effective external potential. The 
final expression for this method is:

.. math::

  v_{xc}(\mathbf{r})=\bar{\epsilon}^{KS}(\mathbf{r}) +
  \frac{\nabla^2 n_{in}(\mathbf{r})}{4n_{in}(\mathbf{r})} 
  - \frac{|\nabla n_{in}(\mathbf{r})|^2}{8|n_{in}(\mathbf{r})|^2}
  -\frac{\tau^{KS}_P(\mathbf{r})}{n^{KS}(\mathbf{r})}
  -\tilde{v}_{ext}(\mathbf{r}) - v_{H,in}(\mathbf{r}).



And we can generate our sample calculation as:

.. code-block:: python

    Ne = psi4.geometry( 
    """ 
    Ne
    noreorient
    nocom
    units bohr
    symmetry c1
    """ )

    psi4.set_options({"reference" : "rhf",
                    'DFT_SPHERICAL_POINTS': 350,  # Usually specify the DFT spherical grid is highly recommended.
                    'DFT_RADIAL_POINTS': 210,  # See [https://psicode.org/psi4manual/master/dft.html] for options.
                    'CUBIC_BASIS_TOLERANCE': 1e-21, 
                    'DFT_BASIS_TOLERANCE': 1e-21, 
                    })  # Spin-Restricted

    wfn = psi4.properties("CCSD/cc-pcvqz", return_wfn=True, molecule=Ne, properties=["dipole"])[1]

    ine = n2v.Inverter(wfn)

We will need to introduce a grid to express the potential for visualization

.. code-block:: python

    x = np.linspace(-5,10,1501)
    y = [0]
    z = [0]
    grid, shape = ine.generate_grids(x,y,z)

And we can finally invert the density.

.. code-block:: python

  v = ine.invert("OC", vxc_grid=grid, guide_potential_components=["hartree"], 
               opt_max_iter=35, frac_old=0.9, init="SCAN")

  