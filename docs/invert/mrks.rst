Modified Ryabinkin-Kohut-Staroverov
===================================

The modified Ryabinkin-Kohut-Staroverov (mRKS) method is an accurate method that makes use of the one and two-electron reduced density matrices.  

The full method's description can be found:  
Ospadov, Egor, Ilya G. Ryabinkin, and Viktor N. Staroverov.  
"Improved method for generating exchange-correlation potentials from electronic wave functions."  
The Journal of chemical physics 146.8 (2017): 084103.

The exchange-correlation potential is found self-consistently through the equation:

.. math::
  v_{xc}(\mathbf{r})=v_{xc}^{hole}(\mathbf{r}) + \bar{\epsilon}^{KS}(\mathbf{r}) - \bar{\epsilon}^{WF}(\mathbf{r}) + \frac{\tau^{WF}_P(\mathbf{r})}{n^{WF}(\mathbf{r})} - \frac{\tau^{KS}_P(\mathbf{r})}{n^{KS}(\mathbf{r})}.

Where each of the components is defined as:

.. math::

  &v_{xc}^{hole}(\mathbf{r})=\int d\mathbf{r}_2 \frac{n_{xc}(\mathbf{r}, \mathbf{r}_2)}{|\mathbf{r}-\mathbf{r}_2|},\label{equ:mRKSComponent_a}\\
  &\bar{\epsilon}^{KS}(\mathbf{r})=\frac{2}{n^{KS}(\mathbf{r})}\sum_{i=1}^{N/2}\epsilon_i|\psi_i(\mathbf{r})|^2,\label{equ:mRKSComponent_b}\\
  &\bar{\epsilon}^{WF}(\mathbf{r})=\frac{2}{n^{WF}(\mathbf{r})}\sum_{k=1}^{M}\lambda_k|f_k(\mathbf{r})|^2,\label{equ:mRKSComponent_c}\\
  &\tau^{WF}_P(\mathbf{r}) = \frac{2}{n^{WF}(\mathbf{r})}\sum_{k<l}^{M}n_kn_l|\chi_k(\mathbf{r})\nabla\chi_l(\mathbf{r}) - \chi_l(\mathbf{r})\nabla\chi_k(\mathbf{r})|^2,\label{equ:mRKSComponent_d}\\
  &\tau^{KS}_P(\mathbf{r}) = \frac{2}{n^{KS}(\mathbf{r})}\sum_{i<j}^{M}n_in_j|\psi_i(\mathbf{r})\nabla\psi_j(\mathbf{r}) - \psi_j(\mathbf{r})\nabla\psi_i(\mathbf{r})|^2,\label{equ:mRKSComponent_e}

And 

.. math::
    n_{xc}(\mathbf{r}, \mathbf{r}_2) = \frac{\Gamma(\mathbf{r},\mathbf{r}_2;\mathbf{r},\mathbf{r}_2)}{n(\mathbf{r})} - n(\mathbf{r}_2),

Let us now try to invert a Neon atom's density.

.. code-block:: python

    Ne = psi4.geometry( 
    """ 
    0 1
    Ne
    noreorient
    nocom
    units bohr
    symmetry c1
    """ )

    psi4.set_options({"reference" : "rhf",
                    "opdm": True,
                    "tpdm": True,
                    'DFT_SPHERICAL_POINTS': 50,  # Usually specify the DFT spherical grid is highly recommended.
                    'DFT_RADIAL_POINTS': 50,  # See [https://psicode.org/psi4manual/master/dft.html] for options.
                    })  # Spin-Restricted

    # IMPORTANT NOTE: ONLY psi4.CIWavefunction or RHF is supported.
    wfn = psi4.properties("detci/cc-pcvdz", return_wfn=True, molecule=Ne, properties=["dipole"])[1]

    ine = n2v.Inverter(wfn)

    ine.invert("mRKS", opt_max_iter=10, frac_old=0.8, init="scan")