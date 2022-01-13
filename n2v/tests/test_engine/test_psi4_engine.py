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
    wfn = psi4.energy("scan/aug-cc-pvtz", molecule=Ne, return_wfn=True)[1]

    # Extract data needed for n2v. 
    da, db = np.array(wfn.Da()), np.array( wfn.Db())
    ca, cb = np.array(wfn.Ca_subset("AO", "OCC")) , np.array(wfn.Ca_subset("AO", "OCC"))
    ea, eb = np.array(wfn.epsilon_a()), np.array(wfn.epsilon_b())

    # Initialize inverter object. 
    ine = n2v.Inverter( engine='psi4' )
    ine.set_system( Ne, 'aug-cc-pvtz' )
    ine.nalpha = wfn.nalpha()
    ine.nbeta = wfn.nbeta()
    ine.Dt = [da, db]
    ine.ct = [ca, cb]
    ine.et = [ea, eb]

    return ine

# # Generate grid
# npoints=1001
# x = np.linspace(-5,5,npoints)[:,None]
# y = np.zeros_like(x)
# z = y
# grid = np.concatenate((x,y,z), axis=1).T

def test_zmp(ine):

    ine.invert(method='zmp', guide_components='fermi_amaldi', lambda_list=[750], opt_max_iter=100, opt_tol=1e-5)

    assert np.isclose(  ine.eigvecs_a[:5].all(),  
                        np.array( [-30.70374546,  -1.58893038,  -0.73510124,  -0.73510123, -0.73510123] ).all() ) 
    

def test_wuyang(ine):
    
    ine.invert("WuYang", guide_components="fermi_amaldi")

    assert np.isclose( ine.eigvecs_a[:5].all(),  
                        np.array( [-30.79513854,  -1.60108275,  -0.75109325,  -0.75109317, -0.75109295]).all() )


def test_pedeco(ine):

    ine.invert("PDECO", opt_max_iter=200, guide_components="fermi_amaldi", gtol=1e-6)

    assert np.isclose(  ine.eigvecs_a[:5].all(),  
                        np.array( [-30.80179687,  -1.60285771,  -0.75242195,  -0.7524181 ,-0.75241671] ).all())