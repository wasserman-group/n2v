**On the Grid**
===============

An additional import class in *n2v* is the *Grider class*. This class helps you
represent many quantities and operators on the grid. 

Although points can be any points in space, all the functions on the grider are 
classified in two ways depending on the needs of the user. Each of the grider functions
will take as an argument either 1) a set of points expressed in a numpy array 
of the correct size, or 2) a V_potential object. The former is mainly used to 
plot quantities such as density or orbitals, whereas the second is used to obtain 
spherical points on the grid. This shperical grid is the same used in a normal 
DFT calculation and it is used by several of the methods in *n2v*. 

The availiable quantities on the grid are the following:

* Density
* Molecular Orbitals
* Nuclear external potential
* Exchange-correlation potential (local only)
* Gradient of molecular orbitals
* Laplacian of molcular orbitals 

1. Example on rectangular grid. 
You're interested to compute the density of your restricted system in one dimension. 

.. code-block:: python

  # Having computed your density using psi4
  inverter = n2v.Inverter(wfn, pbs='cc-pvdz')

  # Store the density
  Da = np.array(wfn.Da())
  
  # Generate one dimensional grid
  x = np.linspace(-5,5,100)
  y = [0]
  z = [0]
  grid, shape = inverter.generate_grids(x,y,z)

  # Obtain density on grid
  Da_g = inverter.on_grid_density(Da=Da, grid=grid)

  fig, ax  = plt.subplots()
  ax.plot(x, Da_g)

2. Example on spherical grid. 

.. code-block:: python

  # Having computed your density using psi4
  inverter = n2v.Inverter(wfn, pbs='cc-pvdz')

  # Store the density
  Da = np.array(wfn.Da())

  # Obtain density on grid and operate as intended. 
  Da_g = inverter.on_grid_density(Da=Da, Vpot=wfn.V_potential())

  # A V_potential object will only get generated after doing a DFT calculation. 
  # Thus, you may need to perform an additional calculation if DFT is not being used. 

