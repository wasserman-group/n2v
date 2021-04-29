"""
direct.py
"""

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

class Direct():
    def direct_inversion(self, grid, correction=True):
        """
        Given a set of canonical kohn-sham orbitals and energies,
        constructs vxc by inverting the Kohn-Sham Equations.
        If correction is added, it will be performed according to:
        Removal of Basis-Set Artifacts in Kohn–Sham Potentials Recovered from Electron Densities
        Gaiduk + Ryabinkin + Staroverov
        https://pubs.acs.org/doi/abs/10.1021/ct4004146

        Parameters:
        -----------

        grid: np.ndarray. Shape: (3xnpoints)
            Grid used to compute correction. 
            If None, dft grid is used. 
        correction: bool
            Adds correction for spurious basis set artifacts

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

        wfn = self.wfn

        #Sanity check. 
        try:
            functional = wfn.functional()
        except Exception as ex:
            raise ValueError(f"{ex}. Direct inversion only available for DFT methods. ")
        if functional.name() == 'HF':
            raise ValueError(f"Direct inversion only available for DFT methods. ")
                                    
        #Build components on grid:
        if grid is not None:
            #Use user-defined grid
            orb       = self.on_grid_orbitals(Ca=wfn.Ca().np, Cb=wfn.Cb().np, grid=grid)
            lap       = self.on_grid_lap_phi(Ca=wfn.Ca().np, Cb=wfn.Cb().np, grid=grid)
            vex, vha  = self.on_grid_esp(Da=wfn.Da().np, Db=wfn.Db().np, grid=grid)[:2]
            den       = self.on_grid_density(Da=wfn.Da().np, Db=wfn.Db().np, grid=grid)
            eig_a = wfn.epsilon_a_subset("AO", "ALL").np
            eig_b = wfn.epsilon_b_subset("AO", "ALL").np
            
            #Calculate correction
            if correction is True:
                osc_profile = self.get_basis_set_correction(grid)

            
        else:
            #Use DFT grid
            Vpot = wfn.V_potential()
            orb       = self.on_grid_orbitals(Ca=wfn.Ca().np, Cb=wfn.Cb().np, Vpot=Vpot)
            lap       = self.on_grid_lap_phi(Ca=wfn.Ca().np, Cb=wfn.Cb().np, Vpot=Vpot)
            vex, vha  = self.on_grid_esp(Da=wfn.Da().np, Db=wfn.Db().np, Vpot=Vpot)[:2]
            den       = self.on_grid_density(Da=wfn.Da().np, Db=wfn.Db().np, Vpot=Vpot)
            eig_a = wfn.epsilon_a_subset("AO", "ALL").np

            #Calculate correction
            if correction is True:
                dft_grid = self.generate_dft_grid(Vpot)
                osc_profile = self.get_basis_set_correction(dft_grid)

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

        #Add correction
        if correction is True:
            vxc_inverted -= osc_profile

        self.grid.vxc = vxc_inverted
        return vxc_inverted