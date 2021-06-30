import n2v
import numpy as np
import psi4
import pytest

psi4.set_options({"save_jk" : True})
psi4.set_memory(int(2.50e9))

psi4.set_options({
                  'DFT_SPHERICAL_POINTS': 50, 
                  'DFT_RADIAL_POINTS': 50, 
                  "opdm": True,
                  "tpdm": True,
                 })

@pytest.fixture
def wfn():
    Ne = psi4.geometry( 
    """ 
    0 1
    Ne 0.0 0.0 0.0
    noreorient
    nocom
    units bohr
    symmetry c1
    """ )
    psi4.set_options({"reference" : "rhf"}) 
    wfn = psi4.properties("detci/cc-pcvdz", return_wfn=True, molecule=Ne, properties=["dipole"])[1]


    return wfn

def test_sucess_inversion(wfn):

    x = np.linspace(-5,10,1501)
    y = [0]
    z = [0]


    ine = n2v.Inverter(wfn)
    grid, shape = ine.generate_grids(x,y,z)
    ine.invert("mRKS", opt_max_iter=30, frac_old=0.8, init="scan")

    correct_eigs_a = np.array([-30.57378503,-1.61491607,-0.75653413,-0.75653413,-0.75653413])

    assert np.isclose(ine.eigvecs_a[:5].all(), correct_eigs_a.all())
