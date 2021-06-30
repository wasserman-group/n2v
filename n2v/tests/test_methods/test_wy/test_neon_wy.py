import n2v
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
    ine.invert("WuYang", guide_potential_components=["fermi_amaldi"])

    assert ine.opt_info.success == True

