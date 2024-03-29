
<p align="center">
<br>
<img src="https://github.com/wasserman-group/n2v/blob/main/media/logo_png.png" alt="n2v" height=300> <br><br>
<a href="https://github.com/wasserman-group/n2v/actions"> <img src="https://github.com/wasserman-group/n2v/actions/workflows/CI.yaml/badge.svg" /></a>
<a href="https://lgtm.com/projects/g/wasserman-group/n2v/?mode=list"><img src="https://img.shields.io/lgtm/grade/python/github/wasserman-group/n2v"></a>
<a href="https://codecov.io/gh/wasserman-group/n2v"> <img src="https://codecov.io/gh/wasserman-group/n2v/branch/main/graph/badge.svg?token=4B8r0cQ2Wk" /></a>
<a href="https://github.com/wasserman-group/n2v/blob/main/LICENSE"><img src="https://img.shields.io/github/license/wasserman-group/n2v?color=blue" /></a>
<br>
</p>

  
### Tutorials:
- Learn how to use `n2v` with these [examples](https://github.com/wasserman-group/n2v_examples) 
<!-- - or try it without installing: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/wasserman-group/n2v_examples/HEAD) -->

#### Installation. Additional Dependencies: 
- Psi4 or PySCF must be installed. 
```
conda install -c psi4 psi4
```
```
pip install pyscf
```
- If using pyscf, please install gbasis. 
```
git clone https://github.com/theochem/gbasis.git
cd gbasis
pip install .
```
- Pylibxc must be installed as well. 
```
pip install pylibxc2
```

### Installation: 
```
git clone https://github.com/wasserman-group/n2v.git
cd n2v
pip install .
```

### Additional Information: 
- We recommend the use of a conda environment (Python 3.7 or higher).
- If installing in Windows, we recommend the use of [WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10).
- If any unexpected error occurs, please contact Victor H. Chavez or Yuming Shi at gonza445@purdue.edu or shi449@purdue.edu respectively. 

### Copyright
Copyright (c) 2021, Wasserman Group  

#### Acknowledgements
*Victor H. Chavez* was supported by a fellowship from The Molecular Sciences Software Institute under NSF grant OAC-1547580.  
Project based on the [MolSSI Cookiecutter](https://github.com/molssi/cookiecutter-cms).  

