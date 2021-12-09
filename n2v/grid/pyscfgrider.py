"""
grider_pyscf.py
Grider for PyScf
"""

from gbasis.evals.density import evaluate_density
from gbasis.evals.eval import evaluate_basis
from gbasis.evals.electrostatic_potential import (point_charge_integral, 
     electrostatic_potential)
from gbasis.wrappers import from_pyscf

from grid.molgrid import MolGrid
from grid.onedgrid import GaussLaguerre,  GaussLegendre, HortonLinear
from grid.becke import BeckeWeights

import numpy as np
from opt_einsum import contract

class PySCFGrider:
    def __init__(self, mol):

        self.mol   = mol
        self.basis = from_pyscf(mol)
        self.atomic_charges = self.mol.atom_charges()
        self.atomic_coords  = self.mol.atom_coords()
        
        # Build spherical grid
        rad          = GaussLaguerre(70)
        becke = BeckeWeights(order=3)
        grid = MolGrid.from_preset( self.atomic_charges,
                                    self.atomic_coords, 
                                    rad,
                                    ['fine' for i in range(len(self.atomic_charges))],
                                    becke )
        
        self.spherical_points = grid.points
        self.w                = grid.weights

        # Build rectangular grid
        self.rectangular_grid   = None

    def build_rectangular(self, npoints, overage=3):
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

        if grid == 'spherical':
            points = self.spherical_points
        elif grid == 'rectangular':
            assert self.rectangular_grid is not None, "Rectangular Grid must be defined first"
            points = self.rectangular_grid
        else: 
            raise ValueError("Specify either spherical or rectangular grid")

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
        if grid == 'spherical':
            points = self.spherical_points
        elif grid == 'rectangular':
            assert self.rectangular_grid is not None, "Rectangular Grid must be defined first"
            points = self.rectangular_grid
        else: 
            raise ValueError("Specify either spherical or rectangular grid")

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

        """        
        if grid == 'spherical':
            points = self.spherical_points
        elif grid == 'rectangular':
            assert self.rectangular_grid is not None, "Rectangular Grid must be defined first"
            points = self.rectangular_grid
        else: 
            raise ValueError("Specify either spherical or rectangular grid")
       
        old_settings = np.seterr(divide="ignore")  # silence warning for dividing by zero
        external_potential = self.atomic_charges[None, :] \
        / (np.sum((points[:, :, None] - self.atomic_coords.T[None, :, :]) ** 2, axis=1) ** 0.5)
    
        return -external_potential

    def to_ao(self, mat_nm, grid='spherical'):
        """
        Takes quantity back to AO orbital basis

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
        mat_g: np.ndarray
            Vector/Matrix expressed on the requested grid
        """
        
        if grid == 'spherical':
            points = self.spherical_points
        elif grid == 'rectangular':
            assert self.rectangular_grid is not None, "Rectangular Grid must be defined first"
            points = self.rectangular_grid
        else: 
            raise ValueError("Specify either spherical or rectangular grid")

        phis = evaluate_basis(self.basis, points)
        mat_g = mat_nm.dot(phis)
        if mat_nm.ndim == 2:
            mat_g *= phis

        return mat_g