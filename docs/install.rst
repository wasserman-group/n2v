**Installation**
================

Thank you for choosing to install *n2v*. Here we suggest three different options
to install *n2v* with its psi4 interface. 

The easiest way to install is through pip. Notice that the package name is *ntov*
But you will still import module as *n2v*.

.. code-block:: bash

  pip install ntov

Be mindful that the pip installation may not include all of the latest changes, 
thus in case the version obtained through pip is rather old, you should 
downloading and installing manually:

.. code-block:: bash

  git clone https://github.com/wasserman-group/n2v.git
  cd n2v
  pip install . 

If you don't have *libxc* it must be installed as well. Notice that we require
the python bindings as well. Let us first install libxc:

.. code-block:: bash

  conda install -c conda-forge libxc

And we communicate libxc with your python site-packages folder:

.. code-block:: bash

  wget http://www.tddft.org/programs/libxc/down.php?file=5.0.0/libxc-5.0.0.tar.gz
  tar -xf libxc-5.0.0.tar.gz
  cd libxc-5.0.0
  python setup.py install


**Additional Information:**

- We recommend the use of a conda environment (<3.7).  

- If installing in Windows, we recommend the use of WSL.  

- If any unexpected error occurs, please contact us at: gonza445@purdue.edu





