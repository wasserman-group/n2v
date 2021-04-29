"""
Basis Set Artifact Correction. 
"""

import numpy as np
from warnings import warn
import psi4

def invert_kohn_sham_equations(self, wfn, grid):
    """
    Given a set of canonical kohn-sham orbitals and energies,
    construct vxc by inverting the Kohn-Sham Equations.
    Removal of Basis-Set Artifacts in Kohn–Sham Potentials Recovered from Electron Densities
    Gaiduk + Ryabinkin + Staroverov
    https://pubs.acs.org/doi/abs/10.1021/ct4004146

    Parameters:
    -----------
    wfn: psi4.core.Wavefunction
        Wavefunction from target calculation. 
        Must be obtained from same basis set if future correction will be applied
    grid: np.ndarray. Shape: (3xnpoints)
        Grid used to compute correction. 
        If None, dft grid is used. 

    Returns:
    --------
    vxc_inverted: np.ndarray
        Vxc obtained from inverting kohn sham equations. 
        Equation (3)
    vxc_lda: np.ndarray
        LDA potential from forward calculation. 
    osc_profile: np.ndarray
        Oscillatory profile that corrects inverted potentials. 
        Equation (5)
    """

    if wfn.basisset().name() != self.basis_str:
        warn("""Basis set from calculation is different from Inverter object. \n
                Addition of correction won't fix basis set artifacts.""")
                                
    #Build components on grid:
    if grid is not None:
        #Use user-defined grid
        orb       = self.on_grid_orbitals(Ca=wfn.Ca().np, Cb=wfn.Cb().np, grid=grid)
        lap       = self.on_grid_lap_phi(Ca=wfn.Ca().np, Cb=wfn.Cb().np, grid=grid)
        vex, vha  = self.on_grid_esp(Da=wfn.Da().np, Db=wfn.Db().np, grid=grid)[:2]
        den       = self.on_grid_density(Da=wfn.Da().np, Db=wfn.Db().np, grid=grid)
        eig_a = wfn.epsilon_a_subset("AO", "ALL").np
        eig_b = wfn.epsilon_b_subset("AO", "ALL").np
        
    else:
        #Use DFT grid
        Vpot = wfn.V_potential()
        orb       = self.on_grid_orbitals(Ca=wfn.Ca().np, Cb=wfn.Cb().np, Vpot=Vpot)
        lap       = self.on_grid_lap_phi(Ca=wfn.Ca().np, Cb=wfn.Cb().np, Vpot=Vpot)
        vex, vha  = self.on_grid_esp(Da=wfn.Da().np, Db=wfn.Db().np, Vpot=Vpot)[:2]
        den       = self.on_grid_density(Da=wfn.Da().np, Db=wfn.Db().np, Vpot=Vpot)
        eig_a = wfn.epsilon_a_subset("AO", "ALL").np

    #Build Reversed LDA from orbitals and density
    kinetic = np.zeros_like(den)
    propios = np.zeros_like(den) 

    #Build v_eff. equation (2) for restricted and unrestricted
    if self.ref == 1:
        for i in range(wfn.nalpha()):
            #Alpha orbitals
            kinetic += 2.0 * (0.5 * orb[i] * lap[i])
            propios += 2.0 * eig_a[i] * np.abs(orb[i])**2 
        
        with np.errstate(divide='ignore', invalid='ignore'):
            veff = (kinetic + propios) / den
        vxc_inverted = veff - vha - vex
        
    elif self.ref== 2:
        for i in range(wfn.nalpha()):
            #Alpha orbitals
            kinetic[:,0] += (0.5 * orb[i][:,0] * lap[i][:,0])
            propios[:,0] += eig_a[i] * np.abs(orb[i][:,0])**2 
        for i in range(wfn.nbeta()):
            #Beta orbitals
            kinetic[:,1] += (0.5 * orb[i][:,1] * lap[i][:,1])
            propios[:,1] += eig_b[i] * np.abs(orb[i][:,1])**2 

        with np.errstate(divide='ignore', invalid='ignore'):
            veff = (kinetic + propios) / den
        vxc_inverted = veff - np.repeat(vha[:,None], 2, axis=1) - np.repeat(vex[:,None], 2, axis=1)

    return vxc_inverted

def basis_set_correction(self, grid):
    """
    Obtains basis set correction for inverted potentials as in:
    Removal of Basis-Set Artifacts in Kohn–Sham Potentials Recovered from Electron Densities
    Gaiduk + Ryabinkin + Staroverov
    https://pubs.acs.org/doi/abs/10.1021/ct4004146

    Parameters:
    -----------
    grid: np.ndarray. Shape: (3xnpoints)
        Grid used to compute correction. 
        If None, dft grid is used. 

    Returns:
    --------
    vxc_inverted: np.ndarray
        Vxc obtained from inverting kohn sham equations. 
        Equation (3)
    vxc_lda: np.ndarray
        LDA potential from forward calculation. 
    osc_profile: np.ndarray
        Oscillatory profile that corrects inverted potentials. 
        Equation (5)
    """
    
    #Make a calculation with LDA.
    _, correction_wfn = psi4.energy("svwn/"+self.basis_str, return_wfn=True, molecule=self.mol)
    
    #Build components on grid:
    if grid is not None:
        #Use user-defined grid
        orb       = self.on_grid_orbitals(Ca=correction_wfn.Ca().np, Cb=correction_wfn.Cb().np, grid=grid)
        lap       = self.on_grid_lap_phi(Ca=correction_wfn.Ca().np, Cb=correction_wfn.Cb().np, grid=grid)
        vex, vha  = self.on_grid_esp(Da=correction_wfn.Da().np, Db=correction_wfn.Db().np, grid=grid)[:2]
        vxc_lda   = self.on_grid_vxc(Da=correction_wfn.Da().np, Db=correction_wfn.Db().np, grid=grid)
        den       = self.on_grid_density(Da=correction_wfn.Da().np, Db=correction_wfn.Db().np, grid=grid)
        eig_a = correction_wfn.epsilon_a_subset("AO", "ALL").np
        eig_b = correction_wfn.epsilon_b_subset("AO", "ALL").np
        
    else:
        #Use DFT grid from LDA calculation
        Vpot = correction_wfn.V_potential()
        orb       = self.on_grid_orbitals(Ca=correction_wfn.Ca().np, Cb=correction_wfn.Cb().np, Vpot=Vpot)
        lap       = self.on_grid_lap_phi(Ca=correction_wfn.Ca().np, Cb=correction_wfn.Cb().np, Vpot=Vpot)
        vex, vha  = self.on_grid_esp(Da=correction_wfn.Da().np, Db=correction_wfn.Db().np, Vpot=Vpot)[:2]
        vxc_lda   = self.on_grid_vxc(Da=correction_wfn.Da().np, Db=correction_wfn.Db().np, Vpot=Vpot)
        den       = self.on_grid_density(Da=correction_wfn.Da().np, Db=correction_wfn.Db().np, Vpot=Vpot)
        eig_a = correction_wfn.epsilon_a_subset("AO", "ALL").np
    
    #Build Reversed LDA from orbitals and density
    kinetic = np.zeros_like(den)
    propios = np.zeros_like(den) 

    #Build v_eff. equation (2) for restricted and unrestricted
    if self.ref == 1:
        for i in range(correction_wfn.nalpha()):
            #Alpha orbitals
            kinetic += 2.0 * (0.5 * orb[i] * lap[i])
            propios += 2.0 * eig_a[i] * np.abs(orb[i])**2 
        
        with np.errstate(divide='ignore', invalid='ignore'):
            veff = (kinetic + propios) / den
        vxc_inverted = veff - vha - vex
        osc_profile = vxc_inverted - vxc_lda
        
    elif self.ref== 2:
        for i in range(correction_wfn.nalpha()):
            #Alpha orbitals
            kinetic[:,0] += (0.5 * orb[i][:,0] * lap[i][:,0])
            propios[:,0] += eig_a[i] * np.abs(orb[i][:,0])**2 
        for i in range(correction_wfn.nbeta()):
            #Beta orbitals
            kinetic[:,1] += (0.5 * orb[i][:,1] * lap[i][:,1])
            propios[:,1] += eig_b[i] * np.abs(orb[i][:,1])**2 

        with np.errstate(divide='ignore', invalid='ignore'):
            veff = (kinetic + propios) / den
        vxc_inverted = veff - np.repeat(vha[:,None], 2, axis=1) - np.repeat(vex[:,None], 2, axis=1)
        osc_profile = vxc_inverted - vxc_lda

    return osc_profile