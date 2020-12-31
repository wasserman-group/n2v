"""
grider.py

Generates grid for plotting 
"""

import numpy as np
from dataclasses import dataclass
import psi4
psi4.core.be_quiet()

from .cubeprop import Cubeprop

import matplotlib.pyplot as plt
from opt_einsum import contract

@dataclass
class data_bucket:
    pass

class Grider(Cubeprop):

    def build_rectangular_grid(self, 
                               L = [3.0, 3.0, 3.0], 
                               D = [0.1, 0.1, 0.1]):

        psi4.set_options({"CUBIC_BLOCK_MAX_POINTS" : 1000000})
                    
        O, N = self.build_grid(L, D)
        block, points, nxyz, npoints, grid =  self.populate_grid(O, N, D)
        # return block, points, nxyz, npoints, [x_plot, y_plot, z_plot]

        # GENERATE DENSITY
        density = np.zeros(int(npoints))
        points.set_pointers(psi4.core.Matrix.from_array(self.Da))
        rho = points.point_values()["RHO_A"]
        offset = 0
        for i in range(len(block)):
            points.compute_points(block[i])
            n_points = block[i].npoints()
            offset += n_points
            density[offset-n_points:offset] = 0.5 * rho.np[:n_points]
        cube_density = np.reshape(density, (int(N[0]), int(N[1]), int(N[2])))

        self.grid = data_bucket
        self.grid.x, self.grid.y, self.grid.z = grid[0], grid[1], grid[2]
        self.grid.density = cube_density
        self.grid.full_dnesity = density 


        #GENERATE HARTREE


        #Hartree

        #XC

        #From Optz

        #Density

        psi4.set_options({"CUBIC_BLOCK_MAX_POINTS" : 1000})

        return grid, density, N
        

    def get_from_grid(self):
    
        mol_grid  = self.mol
        basis_str = self.basis_str
        density_a = self.Da
        density_b = self.Db

        _, wfn = psi4.energy( "svwn/"+basis_str, molecule=mol_grid, return_wfn=True)
        da_p4 = psi4.core.Matrix.from_array( density_a )
        db_p4 = psi4.core.Matrix.from_array( density_b )
        wfn.V_potential().set_D([ da_p4, db_p4 ])
        wfn.V_potential().properties()[0].set_pointers( da_p4, db_p4 )

        natoms = wfn.nalpha() + wfn.nbeta()
        vpot = wfn.V_potential()
        points = vpot.properties()[0]
        functional = vpot.functional()
        results_grid = data_bucket

        if True:
            mol_dict = wfn.molecule().to_schema(dtype='psi4')
            natoms = len(mol_dict["elem"])
            indx = [i for i in range(natoms) if wfn.molecule().charge(i) != 0.0]
            natoms = len(indx)
            #Atomic numbers and Atomic positions
            zs = [mol_dict["elez"][i] for i in indx]
            rs = [wfn.molecule().geometry().np[i] for i in indx]

        vext_block, vha_block, vxc_a_block, vxc_b_block = [], [], [], []
        vinv_a_block, vinv_b_block = [], []
        xb, yb, zb, zz, vha_z, vxc_az, vxc_bz = [], [], [], [], [], [], []
        vinv_az, vinv_bz = [], []
        xc_e       = 0.0

        for b in range(vpot.nblocks()):

            block = vpot.get_block(b)
            points.compute_points(block)
            npoints = block.npoints()
            lpos = np.array( block.functions_local_to_global() )
            phi = np.array( points.basis_values()["PHI"])[:npoints, :lpos.shape[0]]

            x, y, z, w = np.array(block.x()), np.array(block.y()), np.array(block.z()), np.array(block.w())
            xb.append(x) ; yb.append(y) ; zb.append(z)

            if True: #External
                vext_single = np.zeros(npoints)
                for atom in range(natoms):
                    vext_single += -1.0 * zs[atom] / np.sqrt( (x-rs[atom][0])**2 
                                                            + (y-rs[atom][1])**2
                                                            + (z-rs[atom][2])**2)
                vext_block.append(vext_single)
            if True: #ESP

                grid_block = np.array((x,y,z)).T    
                esp = psi4.core.ESPPropCalc(wfn)
                grid_block = psi4.core.Matrix.from_array(grid_block)
                esp_block = esp.compute_esp_over_grid_in_memory(grid_block).np

                # mol = gto.M(atom='H',
                #             spin=1,
                #             basis=basis)
                # esp_block = eval_vh(mol, grid_block, density_a + density_b )
                vha_block.append(-1.0 * (esp_block + vext_single))

            if True: #RHO / VXC
                if density_a is not None and density_b is None:
                    lD = density_a[(lpos[:, None], lpos)]
                    rho = 2.0 * contract('pm,mn,pn->p', phi, lD, phi) 
                    inp = {}
                    inp["RHO_A"] = psi4.core.Vector.from_array(rho)
                    ret = functional.compute_functional(inp, -1)
                    vk   = np.array(ret["V"])[:npoints]
                    xc_e += np.einsum('a,a', w, vk, optimize=True)

                elif density_a is not None and density_b is not None:
                    lDa  = density_a[(lpos[:, None], lpos)]
                    lDb  = density_b[(lpos[:, None], lpos)]
                    lva  = self.v[:self.naux][lpos]
                    lvb  = self.v[:self.naux][lpos]
                    rho_a = contract('pm,mn,pn->p', phi, lDa, phi)
                    rho_b = contract('pm,mn,pn->p', phi, lDb, phi) 
                    inp = {}
                    inp["RHO_A"] = psi4.core.Vector.from_array(rho_a)
                    inp["RHO_B"] = psi4.core.Vector.from_array(rho_b)
                    ret = functional.compute_functional(inp, -1)
                    vk   = np.array(ret["V"])[:npoints]
                    vxc_a_block.append(np.array(ret["V_RHO_A"])[:npoints])
                    vxc_b_block.append(np.array(ret["V_RHO_B"])[:npoints])
                    xc_e += np.einsum('a,a', w, vk, optimize=True)
            
            # if True: #INVERTED COMPONENT
            #         vinv_a_block.append( contract('pm,m->', phi, lva))
            #         vinv_b_block.append( contract('pm,m->', phi, lvb))

        #Save ordered grid
        x = np.concatenate( [i for i in xb] );  y = np.concatenate( [i for i in yb] ); z = np.concatenate( [i for i in zb] )
        indx = np.argsort(z)
        x,y,z = x[indx], y[indx], z[indx]
        results_grid.x, results_grid.y, results_grid.z = x, y, z
        #Save Exc
        results_grid.exc = float(xc_e) 
        #Save VXC
        vxc_a, vxc_b   = np.concatenate( [i for i in vxc_a_block] ), np.concatenate( [i for i in vxc_b_block] )
        #vinv_a, vinv_b = np.concatenate([i for i in vinv_a_block]), np.concatenate([i for i in vinv_b_block])
        vxc_a, vxc_b   = vxc_a[indx], vxc_b[indx]
        #vinv_a, vinv_b = vinv_a[indx], vinv_b[indx]
        results_grid.vxc_a,  results_grid.vxc_b  = vxc_a, vxc_b
        #results_grid.vinv_a, results_grid.vinv_b = vinv_a, vinv_b
        #Save VHartree
        vha = np.concatenate( [i for i in vha_block] )
        vha = vha[indx]        
        results_grid.vha = vha
        #Save Results along Z axis
        for i in range(len(x)):
            if np.abs(x[i]) < 1e-11:
                if np.abs(y[i]) < 1e-11:
                    zz.append(z[i])
                    vha_z.append(vha[i])
                    vxc_az.append(vxc_a[i])
                    vxc_bz.append(vxc_b[i])
                    # vinv_az.append(vinv_a[i])
                    # vinv_bz.append(vinv_b[i])

        results_grid.zz     = zz
        results_grid.vha_z  = vha_z
        results_grid.vxc_az = vxc_az
        results_grid.vxc_bz = vxc_bz
        # results_grid.vinv_az = vinv_az
        # results_grid.vinv_bz = vinv_bz
 
        self.on_grid = results_grid
 



