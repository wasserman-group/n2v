PDE Constrained Optimization
============================

Taking a similar approach as the Wu and Yang method, the PDE-Constrained Optimization (PDECO) method,
is refined by defining a density error so that the Lagrangian is optimized under several constraints. 

More information on this methodology can be found at:

Jensen, Daniel S., and Adam Wasserman.  
"Numerical methods for the inverse problem of density functional theory."   
International Journal of Quantum Chemistry 118.1 (2018): e25425.  

and  

Kanungo, Bikash, Paul M. Zimmerman, and Vikram Gavini.  
"Exact exchange-correlation potentials from ground-state electron densities."  
Nature communications 10.1 (2019): 1-9.  


The lagrangian is defined as:

.. math:: 
      &L[v_{KS}, \{\psi_i\}, \{\epsilon_i\}, \{p_i\}, \{\mu_i\}]\\
    =& \int(n(\mathbf{r})-n_{in}(\mathbf{r}))^w d\mathbf{r} \\ 
    & + \sum_{i=1}^{N/2}\int p_i(\mathbf{r})(-\frac{1}{2}\nabla^2+v_{KS}(\mathbf{r}) - \epsilon_i)\psi_i(\mathbf{r})\mathbf{dr}\\
    &+\sum_{i=1}^{N/2}\mu_i(\int|\psi_i(\mathbf{r})|^2\mathbf{dr}-1),

Where the set of p's and mu's are langange multipliers for the constraints. 
And a similar procedure to the Wu Yang method is followed to generate the gradient and hessians to optimize the lagrangian. 