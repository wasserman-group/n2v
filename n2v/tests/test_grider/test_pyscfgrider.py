import n2v 
import psi4
import pytest
from pyscf import gto
import numpy as np
equal = np.allclose

@pytest.fixture
def ine():
    Ne = gto.M(atom = """
                    Ne
                    """,
    basis = 'cc-pvdz')
    # Perform Calculation
    mf = Ne.KS()
    mf.xc = 'svwn'
    mf.kernel()
    # Initialize inverter object. 
    ine = n2v.Inverter( engine='pyscf' )
    # Extract data for n2v. 
    da, db = mf.make_rdm1()/2, mf.make_rdm1()/2
    ca, cb = mf.mo_coeff[:,:Ne.nelec[0]], mf.mo_coeff[:, :Ne.nelec[1]]
    ea, eb = mf.mo_energy, mf.mo_energy
    ine.set_system( Ne, 'cc-pvdz' )
    ine.Dt = [da, db]
    ine.ct = [ca, cb]
    ine.et = [ea, eb]
    x = np.linspace(-5,5,9)[:,None]
    y = np.zeros_like(x)
    z = y
    grid = np.concatenate((x,y,z), axis=1)
    ine.eng.grid.rectangular_grid = grid
    return ine

def test_density(ine):

    d = ine.eng.grid.density(Da=ine.Dt[0], grid='rectangular')
    assert equal(d, 
    np.array([5.46947034e-10, 3.93919234e-06, 1.60358808e-03, 9.67176398e-02,
       2.96293739e+02, 9.67176398e-02, 1.60358808e-03, 3.93919234e-06,
       5.46947034e-10]), atol=1e-1
    )

def test_esp(ine):
    vh = ine.eng.grid.hartree(2*ine.Dt[0], grid='rectangular')
    vext = ine.eng.grid.external('rectangular')

    assert equal(vext, 
    np.array([-2.        , -2.66666667, -4.        , -8.        ,        -np.inf,
       -8.        , -4.        , -2.66666667, -2.        ])
    ,atol=1e-1)
    assert equal(vh, 
    np.array([ 2.        ,  2.66666404,  3.99738556,  7.6831349 , 31.05817275,
        7.6831349 ,  3.99738556,  2.66666404,  2.        ])
    ,atol=1e-1)

def test_orbs(ine):
    o = ine.eng.grid.orbitals(C=ine.ct[0], grid='rectangular')
    assert equal(o[0], 
    np.array([-1.39463768e-08, -2.86604359e-06, -1.27825132e-04,  5.79544373e-04,
        1.67530714e+01,  5.79544373e-04, -1.27825132e-04, -2.86604359e-06,
       -1.39463768e-08])
    ,atol=1e-1)

def test_density_deriv(ine):
    lap = ine.eng.grid.laplacian_density(density=ine.Dt[0], grid='rectangular')
    grad = ine.eng.grid.gradient_density(density=ine.Dt[0], grid='rectangular')
    grad = np.sum(grad, axis=1)
    assert equal(lap,
    np.array([ 3.43338354e-08,  1.20383165e-04,  1.39784110e-02,  7.16792419e-01,
       -4.47271834e+06,  7.16792419e-01,  1.39784110e-02,  1.20383165e-04,
        3.43338354e-08])
    ,atol=1e-1)
    assert equal(grad,
    np.array([ 4.50591653e-09,  2.34973556e-05,  5.86550336e-03,  3.51484828e-01,
        3.75772624e-15, -3.51484828e-01, -5.86550336e-03, -2.34973556e-05,
       -4.50591653e-09]),
    atol=1e-1)
