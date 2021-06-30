import n2v
import numpy as np
import psi4
import pytest

psi4.set_options({"save_jk" : True})
psi4.set_memory(int(2.50e9))


@pytest.fixture
def wfn():
    Ne = psi4.geometry( 
    """ 
    0 1
    Ne
    noreorient
    nocom
    units bohr
    symmetry c1
    """ )
    psi4.set_options({"reference" : "rhf"}) 
    wfn = psi4.properties("svwn/cc-pvdz",  return_wfn=True, molecule=Ne, property=['dipole'])[1]

    return wfn

def test_sucess_inversion(wfn):
    ine = n2v.Inverter(wfn, pbs="cc-pvtz")
    ine.invert("zmp", opt_max_iter=400, opt_tol=1e-7, zmp_mixing=1, 
           lambda_list=[1,10,20,40], guide_potential_components=["fermi_amaldi"])

    correct_eigs_a = np.array([-30.69866047,-1.59925054,-0.75004983,-0.75004983,-0.75004983])

    assert np.isclose(ine.eigvecs_a[:5].all(), correct_eigs_a.all())


