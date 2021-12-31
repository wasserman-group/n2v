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
    wfn = psi4.properties("pbe/6-311G",  return_wfn=True, molecule=Ne)[1]

    return wfn

def test_sucess_inversion(wfn):

    x = np.linspace(0.1,1,5)
    y = [0]
    z = [0]

    ine = n2v.Inverter(wfn)
    grid, shape = ine.generate_grids(x,y,z)
    vxc_inverted = ine.invert("direct", grid=grid, correction=False)
    correct_vxc = np.array([-6.70647038, -2.01804532, -1.36686845, -1.04725066, -0.84145449])

    vxc_inverted_corrected = ine.invert("direct", grid=grid, correction=True)
    correct_vxc_inverted = np.array([-4.52326477, -1.52151704, -1.22883751, -0.98372241, -0.75157181])

    assert np.isclose(vxc_inverted.all(), correct_vxc.all())
    assert np.isclose(vxc_inverted_corrected.all(), correct_vxc_inverted.all())

