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
    target_0 = np.array([-1.39426057e-08, -1.27790509e-04,  1.67530855e+01, -1.27790509e-04, -1.39426057e-08])
    target_1 = np.array([-1.25081898e-06, -1.15539283e-02,  3.95352024e+00, -1.15539283e-02, -1.25081898e-06])

    print(orbitals[0])
    print(orbitals[1])

    assert np.isclose(orbitals[0], target_0, atol=1e-3).all()
    assert np.isclose(orbitals[1], target_1, atol=1e-3).all()

def test_lap_phi( Ne ):

    ine = Ne[0]
    wfn = Ne[1]
    grid = Ne[2]

    lap = ine.on_grid_lap_phi(Ca=wfn.Ca().np, Cb=wfn.Cb().np, grid=grid)
    target_0 = np.array([-2.89807670e-07, -3.40206526e-04, -1.26411748e+05, -3.40206526e-04, -2.89807670e-07])
    target_1 = np.array([-2.59992244e-05, -3.57517732e-02, -3.00612855e+04, -3.57517732e-02,-2.59992244e-05])

    print(lap[0])
    print(lap[1])

    assert np.isclose(lap[0], target_0).all()
    assert np.isclose(lap[1], target_1).all()

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


