import n2v 
from pyscf import gto, scf, lib, dft, ao2mo, df
import pytest
import numpy as np

@pytest.fixture
def ine():
    # Define Molecule
    Ne = gto.M(atom = """
                    Ne
                    """,
    basis = 'aug-cc-pvtz')

    # Perform Calculation
    mf = Ne.KS()
    mf.xc = 'scan'
    mf.kernel()

    # Extract data for n2v. 
    da, db = mf.make_rdm1()/2, mf.make_rdm1()/2
    ca, cb = mf.mo_coeff[:,:Ne.nelec[0]], mf.mo_coeff[:, :Ne.nelec[1]]
    ea, eb = mf.mo_energy, mf.mo_energy

    # Initialize inverter object. 
    ine = n2v.Inverter( engine='pyscf' )

    ine.set_system( Ne, 'aug-cc-pvtz' )
    ine.Dt = [da, db]
    ine.ct = [ca, cb]
    ine.et = [ea, eb]

    return ine

def test_zmp(ine):

    ine.invert(method='zmp', guide_components='fermi_amaldi', lambda_list=[750], opt_max_iter=100, opt_tol=1e-5)

    assert np.isclose(  ine.eigvecs_a[:5].all(),  
                        np.array( [-30.70166724,  -1.58399267,  -0.73369545,  -0.73369545, -0.73369545] ).all() ) 
    

def test_wuyang(ine):
    
    ine.invert("WuYang", guide_components="fermi_amaldi")

    assert np.isclose( ine.eigvecs_a[:5].all(),  
                        np.array( [-30.75604341,  -1.60572171,  -0.75614577,  -0.75614564, -0.75614546] ).all() )


def test_pedeco(ine):

    ine.invert("PDECO", opt_max_iter=200, guide_components="fermi_amaldi", gtol=1e-6)

    assert np.isclose(  ine.eigvecs_a[:5].all(),  
                        np.array( [-30.75843463,  -1.60691725,  -0.75708326,  -0.75708173, -0.75708055] ).all())

def test_oucarter(ine):

    x = np.linspace(-5, 10, 1501)
    y = [0]
    z = [0]
    grid, shape = ine.eng.grid.generate_grid(x, y, z)
    
    with pytest.raises(ValueError):
        ine.invert('OC', vxc_grid=grid)


def test_oucarter(ine):

    x = np.linspace(-5, 10, 1501)
    y = [0]
    z = [0]
    grid, shape = ine.eng.grid.generate_grid(x, y, z)
    
    with pytest.raises(ValueError):
        ine.invert('mRKS', vxc_grid=grid, opt_max_iter=30, frac_old=0.8, init='scan')