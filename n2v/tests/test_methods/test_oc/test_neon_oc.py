import n2v
import numpy as np
import psi4
import pytest

psi4.set_options({"save_jk" : True})
psi4.set_memory(int(2.50e9))

psi4.set_options({
                  'DFT_SPHERICAL_POINTS': 50, 
                  'DFT_RADIAL_POINTS': 50, 
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
    wfn = psi4.properties("cisd/cc-pvtz",  return_wfn=True, molecule=Ne, property=['dipole'])[1]

    return wfn

def test_sucess_inversion(wfn):

    x = np.linspace(-5,10,1501)
    y = [0]
    z = [0]


    ine = n2v.Inverter(wfn)
    grid, shape = ine.generate_grids(x,y,z)
    v = ine.invert("OC", vxc_grid=grid, guide_potential_components=["hartree"], 
                            opt_max_iter=10, frac_old=0.9, init="SCAN")

    correct_eigs_a = np.array([-30.81136297,-1.60657246,-0.76091335,-0.76091335,-0.76091335])

    assert np.isclose(ine.eigvecs_a[:5].all(), correct_eigs_a.all())
