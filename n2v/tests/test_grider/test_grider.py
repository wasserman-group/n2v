import n2v
import psi4
import numpy as np
import pytest

psi4.set_options({"save_jk" : True})
psi4.set_memory(int(2.50e9))


@pytest.fixture
def Ne():
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

    ine = n2v.Inverter(wfn, pbs="cc-pvtz")
    ine.invert("WuYang", guide_potential_components=["fermi_amaldi"])

    x = np.linspace(-5,5,5)
    y = [0]
    z = [0]
    grid, shape = ine.generate_grids(x,y,z)

    return (ine, wfn, grid)

def test_density( Ne ):

    ine = Ne[0]
    wfn = Ne[1]
    grid = Ne[2]

    da = ine.on_grid_density(Da=wfn.Da().np, Db=wfn.Db(), grid=grid)
    target = np.array([1.09245597e-09, 3.20317361e-03, 5.92592390e+02, 3.20317361e-03,1.09245597e-09])

    assert np.isclose(da, target).all()

def test_coeff( Ne ):

    ine = Ne[0]
    wfn = Ne[1]
    grid = Ne[2]

    da = ine.on_grid_ao(coeff=wfn.Da().np, grid=grid)
    target = np.array([5.46227987e-10, 1.60158681e-03, 2.96296195e+02, 1.60158681e-03, 5.46227987e-10])

    assert np.isclose(da, target).all()

def test_coeff_single( Ne ):

    ine = Ne[0]
    wfn = Ne[1]
    grid = Ne[2]

    rest = ine.on_grid_ao(coeff=ine.v_pbs, grid=grid)
    target = np.array([-2.92277751e-07, -2.69322822e-03, -5.15373654e+00, -2.69322822e-03, -2.92277751e-07])

    assert np.isclose(rest, target).all()

def test_esp( Ne ):

    ine = Ne[0]
    wfn = Ne[1]
    grid = Ne[2]

    vext, hartree, v_fa, esp = ine.on_grid_esp(Da=wfn.Da().np, Db=wfn.Db().np, grid=grid)
    t_ext = np.array([ -2., -4.,  0., -4., -2. ])
    t_har = np.array([  2.        ,  3.9973889 , 31.06080946,  3.9973889 ,  2.         ])
    t_fa = np.array([ 1.8       ,  3.59765001, 27.95472852,  3.59765001,  1.8 ])
    t_esp = np.array([ 1.96521466e-10,  2.61109825e-03, -3.10608095e+01,  2.61109825e-03,1.96521466e-10 ])

    assert np.isclose( vext, t_ext ).all()
    assert np.isclose( hartree, t_har ).all()
    assert np.isclose( v_fa, t_fa ).all()
    assert np.isclose( esp, t_esp ).all()

def test_vxc( Ne ):

    ine = Ne[0]
    wfn = Ne[1]
    grid = Ne[2]

    vxc = ine.on_grid_vxc(Da=wfn.Da().np, Db=wfn.Db().np, grid=grid)
    target = np.array([-1.01420351e-03, -1.45161223e-01, -8.27133894e+00, -1.45161223e-01,-1.01420351e-03])

    assert np.isclose(vxc, target).all()


def test_orbitals( Ne ):

    ine = Ne[0]
    wfn = Ne[1]
    grid = Ne[2]

    orbitals = ine.on_grid_orbitals(Ca=wfn.Ca().np, Cb=wfn.Cb().np, grid=grid)
    target_2 = np.array([ -1.19331155e-05, -1.95913453e-02, -1.80092501e-15,  1.95913453e-02, 1.19331155e-05 ])
    target_10 = np.array([ -1.94996884e-20, -2.40318883e-05,  1.27720148e-15, -2.40318883e-05, 2.28806700e-20 ])

    print(orbitals[2])
    print(orbitals[10])

    # assert np.isclose(orbitals[2], target_2, atol=1e-3).all()
    # assert np.isclose(orbitals[10], target_10, atol=1e-3).all()

def test_lap_phi( Ne ):

    ine = Ne[0]
    wfn = Ne[1]
    grid = Ne[2]

    lap = ine.on_grid_lap_phi(Ca=wfn.Ca().np, Cb=wfn.Cb().np, grid=grid)
    target_2 = np.array([ -1.70876116e-04, -9.15497633e-03,  1.01523932e-11,  9.15497633e-03,1.70876116e-04 ])
    target_10 = np.array([ -3.17122488e-19, -2.17229285e-03, -4.75875308e-12, -2.17229285e-03, 2.89742588e-19 ])

    print(lap[2])
    print(lap[10])

    # assert np.isclose(lap[2], target_2).all()
    # assert np.isclose(lap[10], target_10).all()

def test_grad_phi( Ne ):

    ine = Ne[0]
    wfn = Ne[1]
    grid = Ne[2]

    grad = ine.on_grid_grad_phi(Ca=wfn.Ca().np, Cb=wfn.Cb().np, grid=grid)
    target_2 = np.array([ -4.41609823e-05, -1.84240102e-02,  1.47088832e+01, -1.84240102e-02, -4.41609823e-05 ])
    target_10 = np.array([ -8.20775970e-20, -2.26659995e-04, -5.71751099e-15,  2.26659995e-04, -9.48517230e-20 ])

    # assert np.isclose(grad[2], target_2).all()
    # assert np.isclose(grad[10], target_10).all()

def test_int_ao( Ne ):

    ine = Ne[0]
    wfn = Ne[1]
    grid = Ne[2]

    density = ine.on_grid_density(Da=wfn.Da().np, Db=wfn.Db().np, Vpot=wfn.V_potential())
    int_density = ine.dft_grid_to_fock( value=density, Vpot=wfn.V_potential() )

    pass


