n(r) to v(r)
==============================
"Density-to-potential" inversion Suite. 

[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/wasserman-group/n2v/actions/workflows/CI.yaml/badge.svg)](https://github.com/wasserman-group/n2v/actions)
[![LGTM Grade](https://img.shields.io/lgtm/grade/python/github/wasserman-group/n2v)](https://lgtm.com/projects/g/wasserman-group/n2v/?mode=list)
[![codecov](https://codecov.io/gh/VHchavez/n2v/branch/main/graph/badge.svg?token=4B8r0cQ2Wk)](https://codecov.io/gh/VHchavez/n2v)
[![licence](https://img.shields.io/github/license/wasserman-group/n2v?color=blue)](https://github.com/wasserman-group/n2v/blob/main/LICENSE)


### Getting started: 
- Use a Python environment (<3.7).
- Clone and install from this repository.
- If installing in Windows, we recommend the use of [WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10).

```
git clone https://github.com/wasserman-group/n2v.git
cd n2v
pip install . 
```
- Libxc and pylibxc must be installed as well. 
```
conda install -c conda-forge libxc
```
- To communicate libxc with your python site-packages folder (using bash):
```
wget http://www.tddft.org/programs/libxc/down.php?file=5.0.0/libxc-5.0.0.tar.gz
tar -xf libxc-5.0.0.tar.gz
cd libxc-5.0.0
python setup.py install
```

- If any unexpected error occurs, please contact us at: gonza445@purdue.edu 

### Tutorials:
Learn how to use n2v with these [examples](https://github.com/wasserman-group/n2v_examples).  
Or try it [without installing](https://jupyter.org/binder)
  
### Copyright
Copyright (c) 2021, Wasserman Group  

#### Acknowledgements
*Victor H. Chavez* was supported by a fellowship from The Molecular Sciences Software Institute under NSF grant OAC-1547580.  
Project based on the [MolSSI Cookiecutter](https://github.com/molssi/cookiecutter-cms).  