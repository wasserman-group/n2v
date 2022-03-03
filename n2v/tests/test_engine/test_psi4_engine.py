import n2v 
import psi4
import pytest
import numpy as np

@pytest.fixture
def ine():
    Ne = psi4.geometry("""
    0 1
    Ne
    symmetry c1
    units bohr
    """)
    # Perform Calculation
    wfn = psi4.properties("ccsd/aug-cc-pvtz", molecule=Ne, return_wfn=True, properties=['dipole'])[1]
    # Initialize inverter object. 
    ine = n2v.Inverter.from_wfn(wfn)

    return ine

def test_zmp(ine):
    ine.invert("zmp", opt_max_iter=200, opt_tol=1e-7, zmp_mixing=1, 
            lambda_list=np.linspace(10, 1000, 20), guide_components="fermi_amaldi")

    assert np.isclose( ine.eigvecs_a[:5].all(),  
                       np.array([-30.96371089,  -1.64633607,  -0.78780286,  -0.78780286, -0.78780286]).all() ) 
    assert np.isclose( np.diag(ine.Da)[:5].all(), 
                       np.array([0.9520639 , 0.33023202, 0.03466103, 0.13804668, 0.16689393]).all())

    
def test_wuyang(ine):
    
    ine.invert("WuYang", guide_components="fermi_amaldi")

    assert np.isclose( ine.eigvecs_a[:5].all(),  
                        np.array([-30.9112617 ,  -1.59602304,  -0.74201369,  -0.74201369, -0.74201369]).all() )
    assert np.isclose( np.diag(ine.Da)[:5].all(),
                       np.array([0.95298269, 0.32446633, 0.03453696, 0.14332516, 0.16706669]).all() )


def test_pedeco(ine):

    ine.invert("PDECO", opt_max_iter=200, guide_components="fermi_amaldi", gtol=1e-6)

    assert np.isclose( ine.eigvecs_a[:5].all(),  
                       np.array([-30.90838995,  -1.59752788,  -0.74329655,  -0.74328871, -0.74328082]).all() )
    assert np.isclose( np.diag(ine.Da)[:5].all(),
                       np.array([0.95291707, 0.32472977, 0.03449241, 0.1432748 , 0.16698491]).all() )

def test_oucarter(ine):

    x = np.linspace(-5, 10, 1501)
    y = [0]
    z = [0]
    grid, shape = ine.eng.grid.generate_grid(x, y, z)

    v = ine.invert("OC", vxc_grid=grid, opt_max_iter=21, frac_old=0.9, init='SCAN')

    assert np.isclose( ine.eigvecs_a[:5].all(),  
                       np.array([-31.02826629,  -1.7038403 ,  -0.85493085,  -0.85493085, -0.85493085]).all() )
    assert np.isclose( np.diag(ine.Da)[:5].all(),
                       np.array([0.94579632, 0.32844699, 0.03465013, 0.1395006 , 0.16976739]).all() )
    assert np.isclose( v[0].all(), 
                       np.array([-16.83962961,  -1.02552792,  -0.5309617 ,  -0.27295131, -0.23428725]).all() )

def test_mrks(ine):

    mol = ine.eng.mol
    psi4.set_options({'opdm' : True,
                      'tpdm' : True, 
                      'dft_spherical_points' : 50,
                      'dft_radial_points'    : 50, })

    wfn = psi4.properties('detci/cc-pcvdz', return_wfn=True, molecule=mol, properties=['dipole'])[1]
    ine = n2v.Inverter.from_wfn(wfn)
    x = np.linspace(-5, 10, 1501)
    y = [0]
    z = [0]
    grid, shape = ine.eng.grid.generate_grid(x, y, z)
    ine.invert('mRKS', vxc_grid=grid, opt_max_iter=30, frac_old=0.8, init='scan')

    assert np.isclose( ine.eigvecs_a[:5].all(),  
                       np.array([-30.0960936 ,  -1.60549235,  -0.74989818,  -0.74989818, -0.74989818]).all() )
    assert np.isclose( np.diag(ine.Da)[:5].all(),
                       np.array([0.99495901, 0.24875539, 0.34138226, 0.46664006, 0.46664006]).all() )
    assert np.isclose( ine.grid_vxc.all(), 
                       np.array([-7.23643144, -1.05559238, -0.47135156, -0.28189798, -0.20578139]).all() )


