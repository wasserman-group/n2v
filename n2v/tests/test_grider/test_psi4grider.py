import n2v 
import psi4
import pytest
import numpy as np
equal = np.allclose

psi4.set_options({
                  'DFT_SPHERICAL_POINTS': 50, 
                  'DFT_RADIAL_POINTS': 50, 
                 })

@pytest.fixture
def ine():
    Ne = psi4.geometry("""
    Ne
    symmetry c1
    units bohr
    nocom
    noreorient
    """)
    # Perform Calculation
    wfn = psi4.energy("svwn/cc-pvdz", molecule=Ne, return_wfn=True)[1]
    # Initialize inverter object. 
    ine = n2v.Inverter.from_wfn(wfn)
    return ine

def test_density(ine):
    x = np.linspace(-5,5,10)[:,None]
    y = np.zeros_like(x)
    z = y
    grid = np.concatenate((x,y,z), axis=1).T

    d = ine.eng.grid.density(Da=ine.Dt[0], grid=grid)
    assert equal(d, 
    np.array([1.07877771e-09, 3.33203759e-06, 1.06633480e-03, 4.51592058e-02,
    1.99562615e+00, 1.99562615e+00, 4.51592058e-02, 1.06633480e-03,
    3.33203759e-06, 1.07877771e-09]), atol=1e-1
    )

def test_esp(ine):
    x = np.linspace(-5,5,10)[:,None]
    y = np.zeros_like(x)
    z = y
    grid = np.concatenate((x,y,z), axis=1).T

    esp = ine.eng.grid.esp(Da=ine.Dt[0], Db=ine.Dt[1], grid=grid)
    vext = esp[0]; vh = esp[1]

    assert equal(vext, 
    np.array([ -2.        ,  -2.57142857,  -3.6       ,  -6.        ,
    -18.        , -18.        ,  -6.        ,  -3.6       ,
    -2.57142857,  -2.        ])
    ,atol=1e-1)
    assert equal(vh, 
    np.array([ 2.        ,  2.57142754,  3.59931062,  5.92882942, 13.35529895,
    13.35529895,  5.92882942,  3.59931062,  2.57142754,  2.        ])
    ,atol=1e-1)

def test_orbs(ine):
    x = np.linspace(-5,5,10)[:,None]
    y = np.zeros_like(x)
    z = y
    grid = np.concatenate((x,y,z), axis=1).T

    wfn = ine.eng.wfn

    o = ine.eng.grid.orbitals(Ca=wfn.Ca().np, Cb=wfn.Cb().np, grid=grid)
    l = ine.eng.grid.lap_phi(Ca=wfn.Ca().np, Cb=wfn.Cb().np, grid=grid)
    g = ine.eng.grid.grad_phi(Ca=wfn.Ca().np, Cb=wfn.Cb().np, grid=grid)

    assert equal(o[0], 
    np.array([-1.07293063e-08, -1.31536251e-06, -4.84022461e-05, -3.27069565e-04,
        9.32056819e-02,  9.32056819e-02, -3.27069565e-04, -4.84022461e-05,
       -1.31536251e-06, -1.07293063e-08])
    ,atol=1e-1)
    assert equal(l[0], 
    np.array([-2.23016797e-07, -1.50213577e-05, -2.08576873e-04,  4.43726388e-03,
        4.31457311e+00,  4.31457311e+00,  4.43726388e-03, -2.08576873e-04,
       -1.50213577e-05, -2.23016797e-07])
    ,atol=1e-1)
    assert equal(g[0], 
    np.array([-5.22409926e-08, -4.98127521e-06, -1.30541221e-04,  2.83556427e-04,
        8.17083810e-01, -8.17083810e-01, -2.83556427e-04,  1.30541221e-04,
        4.98127521e-06,  5.22409926e-08])
    ,atol=1e-1)