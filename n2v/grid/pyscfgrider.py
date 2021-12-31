"""
grider_pyscf.py
Grider for PyScf
"""

from gbasis.evals.density import (evaluate_density, 
                                  evaluate_posdef_kinetic_energy_density,
                                  evaluate_density_laplacian,
                                  evaluate_density_gradient,
                                  evaluate_general_kinetic_energy_density,
                                )
from gbasis.evals.eval import evaluate_basis
from gbasis.evals.eval_deriv import evaluate_deriv_basis
from gbasis.evals.electrostatic_potential import (point_charge_integral, 
     electrostatic_potential)
from gbasis.wrappers import from_pyscf

# from grid.molgrid import MolGrid
# from grid.onedgrid import GaussLaguerre,  GaussLegendre, HortonLinear
# from grid.becke import BeckeWeights

from pyscf import dft

import numpy as np
from opt_einsum import contract

class PySCFGrider:
    def __init__(self, mol):

        self.mol   = mol
        self.basis = from_pyscf(mol)
        self.atomic_charges = self.mol.atom_charges()
        self.atomic_coords  = self.mol.atom_coords()

        # Perform quick LDA calculation. Generates grid. 
        mf = dft.UKS(self.mol)
        mf.xc = 'svwn'
        mf.kernel()
        self.spherical_points = mf.grids().coords
        self.w                = mf.grids().weights
        self.mf = mf

        # # Build spherical grid using \textit{grid}
        # rad          = GaussLaguerre(70)
        # becke = BeckeWeights(order=3)
        # grid = MolGrid.from_preset( self.atomic_charges,
        #                             self.atomic_coords, 
        #                             rad,
        #                             ['fine' for i in range(len(self.atomic_charges))],
        #                             becke )
        
        # self.spherical_points = grid.points
        # self.w                = grid.weights

        # Build rectangular grid
        self.rectangular_grid   = None

    def assert_grid(self, grid):
        if grid == 'spherical':
            points = self.spherical_points
        elif grid == 'rectangular':
            assert self.rectangular_grid is not None, "Rectangular Grid must be defined first"
            points = self.rectangular_grid
        else: 
            raise ValueError("Specify either spherical or rectangular grid")
    
        return points


    def build_rectangular(self, npoints):
        """
        Builds rectangular grid containing molecule

        Parameters
        ----------
        npoints: tuple
            Number of points per dimension (n_x, n_y, n_z)
        overage: float
            Spacial extent of box

        """

        # xmin, xmax = np.min(self.atomic_coords[:,0])+3, np.max(self.atomic_coords[:,0])+3
        # ymin, ymax = np.min(self.atomic_coords[:,1])+3, np.max(self.atomic_coords[:,0])+3
        # zmin, zmax = np.min(self.atomic_coords[:,2])+3, np.max(self.atomic_coords[:,0])+3
        
        g1 = np.linspace(-10, 10, npoints[0])
        g2 = np.linspace(0, 0, npoints[1])
        g3 = np.linspace(0, 0, npoints[2])
        gx, gy, gz = np.meshgrid(g1, g2, g3)
        g3d = np.vstack( [gx.ravel(), gy.ravel(), gz.ravel()] ).T

        self.x                = g1
        self.y                = g2
        self.z                = g3
        self.rectangular_grid = g3d

    def density(self, density, grid='spherical'):
        """
        Computes density on grid. 

        Parameters
        ----------

        density: np.ndarray.
            Density in AO basis

        grid: str.
            Type of grid used. Default spherical 
            If 'rectangular' used self.rectangular_grid != None 

        Returns
        -------
        density_g: np.ndarray
            Density on the requested grid    
        """

        points = self.assert_grid(grid)

        density_g = evaluate_density(density, self.basis, points)
        return density_g

    def hartree(self, density, grid='spherical'):
        """
        Computes Hartree Potential on grid. 

        Parameters
        ----------

        density: np.ndarray.
            Density in AO basis

        grid: str.
            Type of grid used. Default spherical 
            If 'rectangular' used self.rectangular_grid != None 


        Returns
        -------

        hartree_potential: np.ndarray
            Hartree potential on the requested grid
        """        
        points = self.assert_grid(grid)

        hartree_potential = point_charge_integral(self.basis, 
                                                points, 
                                                -np.ones(points.shape[0]), 
                                                transform=None, 
                                                coord_type='spherical')

        hartree_potential *= density[:, :, None]
        hartree_potential = np.sum(hartree_potential, axis=(0, 1))

        return hartree_potential

    def external(self, grid='spherical'):
        """
        Computes External Potential on grid. 

        Parameters
        ----------
        grid: str
            Type of grid used. Default spherical 
            If 'rectangular' used self.rectangular_grid != None

        Returns
        -------
        external_potential: np.ndarray
            External potential on the given grid. 
        """        
        points = self.assert_grid(grid)       

        old_settings = np.seterr(divide="ignore")  # silence warning for dividing by zero
        external_potential = self.atomic_charges[None, :] \
        / (np.sum((points[:, :, None] - self.atomic_coords.T[None, :, :]) ** 2, axis=1) ** 0.5)
        np.seterr(**old_settings)

        if external_potential.ndim > 1:
            external_potential = np.sum(external_potential, axis=1)

        return -external_potential

    def to_grid(self, f_nm, grid='spherical'):
        """
        Expresses a matrix quantity on the grid

        Parameters
        ----------
        coeff: np.ndarray
            Vector/Matrix on ao basis. 
            Shape: {(num_ao_basis, ), (num_ao_basis, num_ao_basis)}
        grid: str
            Type of grid used. Default spherical 
            If 'rectangular' used self.rectangular_grid != None

        Returns
        -------
        f_g: np.ndarray
            Vector/Matrix expressed on the requested grid
        """
        
        points = self.assert_grid(grid)

        phis = evaluate_basis(self.basis, points)
        f_g = f_nm.dot(phis)
        if f_nm.ndim == 2:
            f_g *= phis

        return f_g

    def to_ao(self, f_g, grid='spherical'):
        """
        Expresses grid quantity on the AO basis
        """

        points = self.assert_grid(grid)

        phis = evaluate_basis(self.basis, points)
        f_nm = contract( 'pb, p,p,pa->ab', phis.T, f_g, self.w, phis.T )
        f_nm = 0.5 * (f_nm + f_nm.T)

        return f_nm
    
    # Specialized for methods. 
    def posdef_kinetic_energy_density(self, density, grid='spherical'):
        """
        Evaluates the positive-definite kinetic energy density on grid
        t = 1/2 \nabla \cdot \nabla \gamma(r,r') 
        """

        points = self.assert_grid(grid)
        t = evaluate_posdef_kinetic_energy_density(density, self.basis, points)

        return t

    def kinetic_energy_density(self, density, alpha=-1/4, grid='spherical'):
        """
        Evaluates the general form of the kinetic energy density
        t = 1/2 \nabla \cdot \nabla \gamma(r,r') + alpha \nabla^2 n(r)
        """

        points = self.assert_grid(grid)
        t = evaluate_general_kinetic_energy_density(density, self.basis, points, alpha=alpha)

        return t

    def kinetic_energy_density_pauli(self, C, grid='spherical', method='grid'):
        """
        Obtains kinetic energy density in terms of the Pauli kinetic energy density
        
        Parameters
        ----------
        C: np.ndarray
            Occupied Molecular Orbitals
        """

        points = self.assert_grid(grid)

        density = C @ C.T
        density_g = self.density(density, grid=grid)

        basis_dx = self.ao_deriv(derivs=[1,0,0], transform=None, grid=grid)
        basis_dy = self.ao_deriv(derivs=[0,1,0], transform=None, grid=grid)
        basis_dz = self.ao_deriv(derivs=[0,0,1], transform=None, grid=grid)

        if method == 'grid':
            orbs = self.orbitals(C, grid=grid)
            d_orbs = ((basis_dx + basis_dy + basis_dz).T @ C).T
            tau_p = np.zeros_like( density_g )
            orbs = self.orbitals(C, grid=grid)
            for i in range(C.shape[1]):
                for j in range(C.shape[1]):
                    if i == j:
                        pass
                    else:
                        tau_p += np.abs( orbs[i,:] * (d_orbs[j,:]) - orbs[j,:] * (d_orbs[i,:]) )**2

        elif method == 'basis':
            basis = self.ao_deriv(grid=grid)
            dx = contract('pm,mi,nj,pn->ijp', basis.T, C, C, basis_dx.T)
            dy = contract('pm,mi,nj,pn->ijp', basis.T, C, C, basis_dy.T)
            dz = contract('pm,mi,nj,pn->ijp', basis.T, C, C, basis_dz.T)
        
            dx = (dx - np.transpose(dx, (1, 0, 2))) ** 2
            dy = (dy - np.transpose(dy, (1, 0, 2))) ** 2
            dz = (dz - np.transpose(dz, (1, 0, 2))) ** 2

            occ = np.ones(C.shape[1])
            occ_matrix = np.expand_dims(occ, axis=0) @ np.expand_dims(occ, axis=1)

            tau_p = np.sum((dx + dy + dz).T * occ_matrix, axis=(1,2)) 


        tau_p /= (2*density_g)

        return tau_p

    def orbitals(self, C, grid='spherical'):
        """
        Obtains orbitals on grid
        """

        points = self.assert_grid(grid)

        phis = evaluate_basis(self.basis, points)
        mat_g = C.T.dot(phis)
        return mat_g

    def ao_deriv(self, derivs=[0,0,0], transform=None, grid='spherical'):
        """ 
        Calculates AO on the grid (and its derivatives). 
        If Transformation is given, e.g. ao2mo, MO will be given
        """

        points = self.assert_grid(grid)

        orbs_deriv = evaluate_deriv_basis( self.basis, points, np.array(derivs), 
                                           transform=transform )

        return orbs_deriv

    def laplacian_density(self, density, grid='spherical'):
        """
        Calculates the laplacian of the density
        """

        points = self.assert_grid(grid)

        lap_density = evaluate_density_laplacian( density, self.basis, points )
        return lap_density
  
    def gradient_density(self, density, grid='spherical'):
        """
        Evaluatges gradient of density on requested grid
        """

        points = self.assert_grid(grid)

        grad_density = evaluate_density_gradient(density, self.basis, points)
        return grad_density

    def avg_local_orb_energy(self, density, orbitals, eigvals, grid='spherical'):
        """
        Generates average local orbital energy. Described by Staroverov
        J. Chem. Phys. 146, 084103. [Equations 4 and/or 6]
        $$
        e_tilde = 1/n(r) * [ \sum_i \varepsilon_i * | \phi_i(r) |^2 ]
        $$
        """

        points = self.assert_grid(grid)

        phis      = evaluate_basis(self.basis, points)
        density_g = self.density(density=density, grid=grid)
        e_tilde   = contract('xp, xo, xo, x, xp-> p', phis, 
                                                      orbitals, orbitals, eigvals, 
                                                      phis) / density_g

        return e_tilde

    def external_tilde(self, grid='spherical', method='grid'):
        """
        Generates effective external potential from LDA exchange. Described by Ou + Carter. 
        J. Chem. Theory Comput. 2018, 14, 11, 5680–5689

        $$
        v^{~}{ext}(r) = \epsilon^{-LDA}(r) - \frac{\tau^{LDA}{L}}{n^{LDA}(r)}
        - v_{H}^{LDA}(r) - v_{xc}^{LDA}(r)
        $$
        (22) in [1].
        """

        points = self.assert_grid(grid)

        # LDA results
        Da0, Db0 = self.mf.make_rdm1()
        Ca0, Cb0 = self.mf.mo_coeff
        ea0, eb0 = self.mf.mo_energy

        da0_g = self.density(Da0, grid)
        db0_g = self.density(Db0, grid)

        # LDA exchange
        cx = -(3/np.pi)**(1/3)
        vxca = cx * da0_g ** (1/3)
        vxcb = cx * db0_g ** (1/3)

        # External tilde
        e_tilde = self.avg_local_orb_energy(Da0, Ca0, ea0, grid=grid)
        lap     = self.laplacian_density(Da0, grid=grid)
        grad    = self.gradient_density(Da0, grid=grid)
        grad    = grad[:,0] + grad[:,1] + grad[:,2]
        hartree = self.hartree(Da0, grid=grid)
        
        tau_l  = self.kinetic_energy_density_pauli(Ca0, grid=grid, method=method)
        tau_l += - 0.25 * lap + np.abs( grad )**2 / (8*da0_g) 
    
        external_tilde = e_tilde - tau_l/da0_g - hartree - vxca

        return external_tilde

