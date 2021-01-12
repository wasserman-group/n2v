# Copyright (c) 2007-2019 The Psi4 Developers.
# Copyright (c) 2014-2018, The Psi4NumPy Developers.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.

#     * Neither the name of the Psi4NumPy Developers nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import psi4

class Cubeprop():

    def build_grid(self, L, D):
        """
        Creates origin and extent of the cube file

        Parameters
        ----------

        wfn : psi4.core.Wavefunction
            Wavefunction object from Psi4 energy calculation
        L : List
            Spacial Extent for x,y,z directions
        D : List
            Grid Spacing in bohrs for x,y,z directions


        Returns
        -------

        O : List
            Origin for the cubefile
        N : List
            Number of points for each coordinate
        """

        geometry = self.mol.full_geometry().np

        Xmin = np.zeros(3)
        Xmax = np.zeros(3)
        Xdel = np.zeros(3)

        N = np.zeros(3)
        O = np.zeros(3)

        for k in [0,1,2]:
            Xmin[k] = Xmax[k] = geometry[0,k]

            for atom in range(len(geometry)):
                Xmin[k] = geometry[atom, k] if Xmin[k] > geometry[atom, k] else Xmin[k]
                Xmax[k] = geometry[atom, k] if Xmax[k] < geometry[atom, k] else Xmax[k]

            Xdel[k] = Xmax[k] - Xmin[k]
            N[k] = int((Xmax[k] - Xmin[k] + 2.0 * L[k]) / D[k])

            if D[k] * N[k] < (Xmax[k] - Xmin[k] + 2.0 * L[k]):
                N[k] += 1

            O[k] = Xmin[k] - (D[k] * N[k] - (Xmax[k] - Xmin[k])) / 2.0

        return O, N

    def populate_grid(self, O, N, D):
        """
        Build cube grid

        Parameters
        ----------

        wfn : psi4.core.Wavefunction
            Wavefunction object from Psi4 energy calculation
        O : List
            Origin for the cubefile
        N : List
            Number of points for each coordinate
        D : List
            Grid Spacing in bohrs for x,y,z directions


        Returns
        -------

        block : List
            Set of psi4.core.BlockOPoints for cube grid
        points : psi4.core.RKSFunctions
        nxyz : integer
            number of points in each direction for rectangular grid
        npoints : int
            total number of points in grid

        """

        epsilon = psi4.core.get_global_option("CUBIC_BASIS_TOLERANCE")
        basis   = psi4.core.BasisSet.build(self.mol, 'ORBITAL', self.basis_str)
        extens  = psi4.core.BasisExtents(basis, epsilon)

        npoints = (N[0]) * (N[1]) * (N[2])

        x = np.zeros(int(npoints), dtype=np.half)
        y = np.zeros(int(npoints), dtype=np.half)
        z = np.zeros(int(npoints), dtype=np.half)
        w = np.zeros(int(npoints), dtype=np.half)

        max_points = psi4.core.get_global_option("CUBIC_BlOCK_MAX_POINTS")
        nxyz = int(np.round(max_points**(1/3)))

        block = []
        offset = 0
        i_start = 0
        j_start = 0
        k_start = 0

        x_plot, y_plot, z_plot = [], [], []

        for i in range(i_start, int(N[0] + 1), nxyz):
            ni = int(N[0]) - i if i + nxyz > N[0] else nxyz
            for j in range(j_start, int(N[1] + 1), nxyz):
                nj = int(N[1]) - j if j + nxyz > N[1] else nxyz
                for k in range(k_start, int(N[2] + 1), nxyz):
                    nk = int(N[2]) - k if k + nxyz > N[2] else nxyz

                    x_in, y_in, z_in, w_in = [], [], [], []

                    block_size = 0
                    for ii in range(i , i + ni):
                        for jj in range(j, j + nj):
                            for kk in range(k, k + nk):

                                x[offset] = O[0] + ii * D[0]
                                y[offset] = O[1] + jj * D[1]
                                z[offset] = O[2] + kk * D[2]
                                w[offset] = D[0] * D[1] * D[2]

                                x_plot.append(x[offset].astype(np.half))
                                y_plot.append(y[offset].astype(np.half))
                                z_plot.append(z[offset].astype(np.half))

                                x_in.append(x[offset])
                                y_in.append(y[offset])
                                z_in.append(z[offset])
                                w_in.append(w[offset])

                                offset     += 1
                                block_size += 1

                    x_out = psi4.core.Vector.from_array(np.array(x_in))
                    y_out = psi4.core.Vector.from_array(np.array(y_in))
                    z_out = psi4.core.Vector.from_array(np.array(z_in))
                    w_out = psi4.core.Vector.from_array(np.array(w_in))

                    block.append(psi4.core.BlockOPoints(x_out, y_out, z_out, w_out, extens))

        max_functions = 0
        for i in range(max_functions, len(block)):
            max_functions = max_functions if max_functions > len(block[i].functions_local_to_global()) else len(block[i].functions_local_to_global())

        points = psi4.core.RKSFunctions(basis, int(npoints), max_functions)
        points.set_ansatz(0)

        x_out = np.array( x_plot )
        y_out = np.array( y_plot )
        z_out = np.array( z_plot )

        return block, points, nxyz, int(npoints), [x_out, y_out, z_out]
        #return block, points, nxyz, npoints

    def add_density(self, npoints, points, block, matrix):
        """
        Computes density in new grid


        Parameters
        ----------

        npoints: int
            total number of points
        points : psi4.core.RKSFunctions
        block : list
            Set of psi4.core.BlockOPoints for cube grid
        matrix : psi4.core.Matrix
            One-particle density matrix


        Returns
        -------

        v : numpy array
            Array with density values on the grid
        """

        v = np.zeros(int(npoints))

        points.set_pointers(matrix)
        rho = points.point_values()["RHO_A"]

        offset = 0
        for i in range(len(block)):
            points.compute_points(block[i])
            n_points = block[i].npoints()
            offset += n_points
            v[offset-n_points:offset] = 0.5 * rho.np[:n_points]

        return v

    def compute_isocontour_range(self, v, npoints):
        """
        Computes threshold for isocontour range

        Parameters
        ----------

        v : numpy array
            Array with scalar values on the grid

        npopints : int
            Total number of points on the grid


        Returns
        -------

        values : list
            Value of positive and negative isocontour

        cumulative_threshold: float

        """
        cumulative_threshold = 0.85

        sum_weight = 0

        #Store the points with their weights and compute the sum of weights
        sorted_points = np.zeros((int(npoints),2))
        for i in range(0, int(npoints)):
            value = v[i]
            weight = np.power(np.abs(value), 1.0)
            sum_weight += weight
            sorted_points[i] = [weight, value]

        #Sort the points
        sorted_points = sorted_points[np.argsort(sorted_points[:,1])][::-1]

        #Determine the positve and negative bounds

        sum = 0

        negative_isocontour = 0.0
        positive_isocontour = 0.0

        for i in range(len(sorted_points)):

            if sorted_points[i,1] >=  0:
                positive_isocontour = sorted_points[i,1]

            if sorted_points[i,1] <  0:
                negative_isocontour = sorted_points[i,1]

            sum += sorted_points[i,0] / sum_weight

            if sum > cumulative_threshold:
                break
        values = [positive_isocontour, negative_isocontour]

        return values, cumulative_threshold

    def write_cube_file(self, wfn, O, N, D, nxyz, npoints, v, name, header):

        #Reorder the grid

        v2 = np.zeros_like(v)

        offset = 0
        for istart in range(0, int(N[0]+1), nxyz):
            ni = int(N[0]) - istart if istart + nxyz > N[0] else nxyz
            for jstart in range(0, int(N[1] + 1), nxyz):
                nj = int(N[1]) - jstart if jstart + nxyz > N[1] else nxyz
                for kstart in range(0, int(N[2] + 1), nxyz):
                    nk = int(N[2]) - kstart if kstart + nxyz > N[2] else nxyz

                    for i in range(istart, istart + ni):
                        for j in range(jstart, jstart + nj):
                            for k in range(kstart, kstart + nk):


                                index = i * (N[1]) * (N[2]) + j * (N[2]) + k
                                v2[int(index)] = v[offset]

                                offset += 1


        f = open(F"./{name}.cube","w+")
        f.write("Psi4Numpy Gaussian Cube File. \n")
        f.write(F"Property: {name}")
        f.write(header)

        f.write(F"{wfn.molecule().natom()}  {format(O[0], '1.5f')}  {format(O[1], '1.5f')}  {format(O[2], '1.5f')} \n")

        f.write(F" {int(N[0])}  {D[0]}  0.0  0.0 \n")
        f.write(F" {int(N[1])}  0.0  {D[1]}  0.0 \n")
        f.write(F" {int(N[2])}  0.0  0.0  {D[2]} \n")

        for atom in range(wfn.molecule().natom()):
            f.write(F"{wfn.molecule().true_atomic_number(atom)}  0.0  {format(wfn.molecule().x(atom), '8.5f')}  {format(wfn.molecule().y(atom), '8.5f')}  {format(wfn.molecule().z(atom), '8.5f')} \n")


        for i in range(int(npoints)):
            f.write(format(v2[i], '1.5e'))
            f.write("  ")
            if i%6 == 5:
                f.write("\n")

        f.close()

    def compute_density(self, wfn   , O, N, D, npoints, points, nxyz, block, matrix, name):

        v = add_density(npoints, points, block, matrix)
        isocontour_range, threshold = compute_isocontour_range(v, npoints)

        density_percent = 100.0 * threshold

        header = F"""[e/a0^3]. Isocontour range for {density_percent} of the density ({format(isocontour_range[0], '1.5f')},{format(isocontour_range[1],'1.5f')}) \n"""

        write_cube_file(wfn, O, N, D, nxyz, npoints, v, name, header)

        return v

    def _getline(self,cube):
        """
        Read a line from cube file where first field is an int
        and the remaining fields are floats.

        Parameters

        ----------
        cube: file object of the cube file

        Returns

        -------
        (int, list<float>)

        """
        l = cube.readline().strip().split()
        return int(l[0]), map(float, l[1:])

    def cube_to_array(self,fname):
        """
        Read cube file into numpy array

        Parameters
        ----------
        fname: filename of cube file

        Returns
        --------
        (data: np.array, metadata: dict)

        """
        meta = {}
        with open(fname, 'r') as cube:
            cube.readline(); cube.readline()  # ignore comments
            natm, meta['org'] = _getline(cube)
            nx, meta['xvec'] = _getline(cube)
            ny, meta['yvec'] = _getline(cube)
            nz, meta['zvec'] = _getline(cube)
            meta['atoms'] = [_getline(cube) for i in range(natm)]
            data = np.zeros((nx*ny*nz))
            idx = 0
            for line in cube:
                for val in line.strip().split():
                    data[idx] = float(val)
                    idx += 1
        data = np.reshape(data, (nx, ny, nz))
        cube.close()
        return data, meta







        