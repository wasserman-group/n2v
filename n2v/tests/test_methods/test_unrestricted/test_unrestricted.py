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
    psi4.set_options({"reference" : "uhf"}) 
    wfn = psi4.properties("svwn/cc-pvdz",  return_wfn=True, molecule=Ne, property=['dipole'])[1]

    return wfn

def test_success_wuyang(wfn):
    ine = n2v.Inverter(wfn, pbs="cc-pvtz")
    ine.invert("WuYang", guide_potential_components=["fermi_amaldi"])

    assert ine.opt_info.success == True
    assert np.isclose( ine.eigvecs_b[:5].all(),
                       np.array([ -30.62095229,  -1.39719357,  -0.594066  ,  -0.594066, -0.594066 ]).all() )
    assert np.isclose( ine.eigvecs_a[:5].all(),
                       np.array([ -30.62095229,  -1.39719357,  -0.594066  ,  -0.594066, -0.594066 ]).all() )


def test_success_zmp(wfn):
    ine = n2v.Inverter(wfn)
    ine.invert("zmp", opt_max_iter=200, opt_tol=1e-7, zmp_mixing=1, 
            lambda_list=[250,500], guide_potential_components=["fermi_amaldi"])

    assert np.isclose( ine.eigvecs_a[:5].all(),
                       np.array([-30.37726896,  -1.38608021,  -0.5762909 ,  -0.5762909 , -0.5762909]).all() )
    assert np.isclose( ine.eigvecs_b[:5].all(),
                       np.array([-30.37726899,  -1.38608021,  -0.5762909 ,  -0.5762909 , -0.5762909]).all() )    


def test_success_pdeco(wfn):
    ine = n2v.Inverter(wfn)
    ine.invert("PDECO", opt_max_iter=200, guide_potential_components=["fermi_amaldi"], gtol=1e-6)

    assert np.isclose( ine.eigvecs_a[:5].all(),
                       np.array([-30.70565268,  -1.42322085,  -0.61265233,  -0.61264822, -0.61264436]).all() )
    assert np.isclose( ine.eigvecs_b[:5].all(),
                       np.array([-30.70565268,  -1.42322085,  -0.61265233,  -0.61264822, -0.61264436]).all() )    


def test_success_oc(wfn):
    ine = n2v.Inverter(wfn)

    x = np.linspace(-5,10,1501) 
    y = [0]
    z = [0]
    grid, shape = ine.generate_grids(x,y,z)

    v = ine.invert("OC", vxc_grid=grid, guide_potential_components=["hartree"], 
                opt_max_iter=35, frac_old=0.9, init="SCAN")

    assert np.isclose( ine.eigvecs_a[:5].all(),
                       np.array([-30.63935848,  -1.31262698,  -0.48222076,  -0.48222076, -0.48222076,]).all() )
    assert np.isclose( ine.eigvecs_b[:5].all(),
                       np.array([-30.63935848,  -1.31262698,  -0.48222076,  -0.48222076, -0.48222076]).all() )    
