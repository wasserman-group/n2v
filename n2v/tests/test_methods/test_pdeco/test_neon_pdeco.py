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
    wfn = psi4.properties("ccsd/aug-cc-pvtz",  return_wfn=True, molecule=Ne, property=['dipole'])[1]

    return wfn

def test_sucess_inversion(wfn):
    ine = n2v.Inverter(wfn, pbs="aug-cc-pvqz")
    ine.invert("PDECO", opt_max_iter=200, guide_potential_components=["fermi_amaldi"], gtol=1e-6)

    correct_eigs_a = np.array([-30.824053,-1.57094445,-0.71606582,-0.71586213,-0.71584401])

    assert np.isclose(ine.eigvecs_a[:5].all(), correct_eigs_a.all())
