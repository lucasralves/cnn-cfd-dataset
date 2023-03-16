"""
    Create meshes
"""
from os import listdir
from typing import List
import numpy as np
from scipy.interpolate import splprep, splev, interp1d
import gmsh
from tqdm import tqdm

class foil_mesh:
    """
        Returns a callable that generate the mesh around a given airfoil.

        Parameters
        ----------
        - ns: Number of points on the profile surface.
        - nt: Number of points on the trailing edge.
        - nf: Number of points of the boundary layer.
        - delta: Boundary layer height.
        - exp1: Expansion ratio of points on the surface towards the leading edge.
        - exp2: Expansion ratio of points on the surface towards the trailing edge.
        - exp3: Boundary layer expansion ratio.
        - ext_radius: outer countor radius.
        - cell_ratio: ratio between the size of the element on the outer counter and on the
          surface of the boundary layer.
    """

    def __init__(self, ns: int = None,
                       nt: int = None,
                       nf: int = None,
                       delta: float = None,
                       exp1: float = None,
                       exp2: float = None,
                       exp3: float = None,
                       ext_radius: float = None,
                       ext_cell_size: float = None) -> None:
        

        self._ns = ns if ns is not None else 100
        self._nt = nt if nt is not None else 6
        self._nf = nf if nf is not None else 50
        self._delta = delta if delta is not None else 0.1
        self._exp1 = exp1 if exp1 is not None else 1.05
        self._exp2 = exp2 if exp2 is not None else 1.05
        self._exp3 = exp3 if exp3 is not None else 1.1
        self._ext_radius = ext_radius if ext_radius is not None else 25
        self._ext_cell_size = ext_cell_size if ext_cell_size is not None else 1

        return
    
    def create(self, file: str,
                     out: str,
                     ns: int = None,
                     nt: int = None,
                     nf: int = None,
                     delta: float = None,
                     exp1: float = None,
                     exp2: float = None,
                     exp3: float = None,
                     ext_radius: float = None,
                     ext_cell_size: float = None) -> None:
        """ Creates a mesh around an airfoil in file and save it to out. """

        # Update parameters
        if ns is not None: self._ns = ns
        if nt is not None: self._nt = nt
        if nf is not None: self._nf = nf
        if delta is not None: self._delta = delta
        if exp1 is not None: self._exp1 = exp1
        if exp2 is not None: self._exp2 = exp2
        if exp3 is not None: self._exp3 = exp3
        if ext_radius is not None: self._ext_radius = ext_radius
        if ext_cell_size is not None: self._ext_cell_size = ext_cell_size

        # Load
        foil = np.loadtxt(file)

        # Redefine
        foil = self._redefine(foil)

        # Add trailing edge points
        foil = self._add_trailing_edge_points(foil)

        # Layers
        X, Y = self._calc_layers(foil)

        # Mesh size points
        sizes_X, sizes_Y, sizes, last_layer_sizes = self._calc_mesh_size_points(X, Y)

        # Mesh
        self._gen_mesh(X, Y, sizes_X, sizes_Y, sizes, last_layer_sizes, out)

        return
    
    def _redefine(self, foil: np.ndarray) -> np.ndarray:
        """Redefine airfoil surface"""

        # Create interpolation
        tck, _ = splprep([foil[:, 0], foil[:, 1]], s=0)

        sum = 0.0
        sum_tot = 0.0
        data = np.zeros(foil.shape[0])

        for i in range(foil.shape[0] - 2):
            data[i + 1] = np.arccos(np.clip(np.dot((foil[i + 1, :] - foil[i, :]) / np.linalg.norm(foil[i + 1, :] - foil[i, :]), (foil[i + 2, :] - foil[i + 1, :]) / np.linalg.norm(foil[i + 2, :] - foil[i + 1, :])), -1.0, 1.0))
        
        arg = int(np.argmax(data))

        for i in range(foil.shape[0] - 1):
            sum_tot = sum_tot + np.linalg.norm(foil[i + 1, :] - foil[i, :])
            if i < arg:
                sum = sum + np.linalg.norm(foil[i + 1, :] - foil[i, :])

        u1 = 0
        u5 = 1
        u3 = sum / sum_tot
        u2 = (u3 + u1) / 2
        u4 = (u5 + u3) / 2

        n = int(self._ns / 4)

        d1 = (u2 - u1) * (self._exp2 - 1) / (self._exp2 ** n - 1)
        d2 = (u3 - u2) * (self._exp1 - 1) / (self._exp1 ** n - 1)
        d3 = (u4 - u3) * (self._exp1 - 1) / (self._exp1 ** n - 1)
        d4 = (u5 - u4) * (self._exp2 - 1) / (self._exp2 ** n - 1)

        c1 = [d1 * (self._exp2 ** (i - 1)) for i in range(n)]
        c2 = [d2 * (self._exp1 ** (i - 1)) for i in range(n)]
        c3 = [d3 * (self._exp1 ** (i - 1)) for i in range(n)]
        c4 = [d4 * (self._exp2 ** (i - 1)) for i in range(n)]

        c2.reverse()
        c4.reverse()

        c = [0]

        for i in range(n):
            c.append(c1[i] + c[i])
        
        for i in range(n):
            c.append(c2[i] + c[n + i])
        
        for i in range(n):
            c.append(c3[i] + c[2 * n + i])
        
        for i in range(n):
            c.append(c4[i] + c[3 * n + i])
        
        c = np.asarray(c)
        c = c / np.max(c)

        tulple_points = splev(c, tck)
        new_points = np.zeros((c.shape[0], 2))
        new_points[:, 0], new_points[:, 1] = np.asarray(list(tulple_points[0])), np.asarray(list(tulple_points[1]))
        
        new_points[:, 0] = new_points[:, 0] - 0.5 * (max(new_points[:, 0]) + min(new_points[:, 0]))
        new_points[:, 1] = new_points[:, 1] - 0.5 * (max(new_points[:, 1]) + min(new_points[:, 1]))

        self._ns = c.shape[0]

        return new_points
    
    def _add_trailing_edge_points(self, foil: np.ndarray) -> np.ndarray:
        """Create the foil trailing edge points"""

        t = (foil[0, :] - foil[self._ns - 1, :]) / np.linalg.norm(foil[0, :] - foil[self._ns - 1, :])
        dist_te = np.linalg.norm(foil[0, :] - foil[self._ns - 1, :]) / (self._nt - 2)
        point_te = np.zeros((self._nt - 3, 2))

        for i in range(self._nt - 3):
            point_te[i, :] = foil[self._ns - 1, :] + t * (i + 1) * dist_te
        
        foil = np.concatenate([foil, point_te], axis=0)

        return foil
    
    def _layer_normals(self, curve: np.ndarray) -> np.ndarray:
        """Create an ndarray containing the normals of the layer"""

        def unary_vector(a: np.ndarray):
            return a / np.linalg.norm(a)

        n = curve.shape[0]

        normals = np.zeros_like(curve)

        for i in range(n):
            if i == 0:
                t1 = unary_vector(curve[i + 1, :] - curve[i, :])
                t2 = unary_vector(curve[i, :] - curve[n - 1, :])
                t = unary_vector(t1 + t2)
                normals[i, 0], normals[i, 1] = t[1], -t[0]
            elif i == n - 1:
                t1 = unary_vector(curve[0, :] - curve[i, :])
                t2 = unary_vector(curve[i, :] - curve[i - 1, :])
                t = unary_vector(t1 + t2)
                normals[i, 0], normals[i, 1] = t[1], -t[0]
            else:
                t1 = unary_vector(curve[i + 1, :] - curve[i, :])
                t2 = unary_vector(curve[i, :] - curve[i - 1, :])
                t = unary_vector(t1 + t2)
                normals[i, 0], normals[i, 1] = t[1], -t[0]
        
        for _ in range(5):
            for i in range(n):
                if i != 0 and i != n - 1:
                    normals[i, :] = unary_vector(normals[i - 1, :] + normals[i + 1, :])
                elif i == 0:
                    normals[i, :] = unary_vector(normals[n - 1, :] + normals[i + 1, :])
                elif i == n - 1:
                    normals[i, :] = unary_vector(normals[i - 1, :] + normals[0, :])

        return normals
    
    def _calc_layers(self, foil) -> List[np.ndarray]:
        """ Create the boundary layer """

        n = 20
        dh = self._delta / (n - 1)
        X = np.zeros((n, foil.shape[0]))
        Y = np.zeros((n, foil.shape[0]))
        layer = np.zeros_like(foil)

        # Surface
        X[0, :] = foil[:, 0]
        Y[0, :] = foil[:, 1]

        for i in range(n - 1):
            layer[:, 0], layer[:, 1] = X[i, :], Y[i, :]
            normals = self._layer_normals(layer)
            X[i + 1, :] = layer[:, 0] + dh * normals[:, 0]
            Y[i + 1, :] = layer[:, 1] + dh * normals[:, 1]
        
        return [X, Y]

    def _calc_mesh_size_points(self, X: np.ndarray, Y: np.ndarray) -> List[np.ndarray]:

        def create_layer(c: np.ndarray, h: float) -> List[np.ndarray]:
            """ Create layer """

            n = 20
            dh = h / (n - 1)
            layer = np.copy(c)

            for _ in range(n - 1):
                normals = self._layer_normals(layer)
                layer[:, 0] = layer[:, 0] + dh * normals[:, 0]
                layer[:, 1] = layer[:, 1] + dh * normals[:, 1]
            
            return layer

        # Refinament layers params
        dist = [0.1, 0.3, 0.5]
        ratio = [2, 5, 100]

        ref_layers = len(dist)
        last_layer = X.shape[0] - 1

        # Data
        curve = np.zeros((X.shape[1], 2))
        curve[:, 0] = X[last_layer, :]
        curve[:, 1] = Y[last_layer, :]

        # Find maximum grid size and last layer sizes
        last_layer_sizes = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            if j == 0:
                last_layer_sizes[j] = 0.5 * ( ( (X[last_layer, j + 1] - X[last_layer, X.shape[1] - 1]) ** 2 + (Y[last_layer, j + 1] - Y[last_layer, X.shape[1] - 1]) ** 2 ) ** 0.5 )
            elif j == X.shape[1] - 1:
                last_layer_sizes[j] = 0.5 * ( ( (X[last_layer, 0] - X[last_layer, j - 1]) ** 2 + (Y[last_layer, 0] - Y[last_layer, j - 1]) ** 2 ) ** 0.5 )
            else:
                last_layer_sizes[j] = 0.5 * ( ( (X[last_layer, j + 1] - X[last_layer, j - 1]) ** 2 + (Y[last_layer, j + 1] - Y[last_layer, j - 1]) ** 2 ) ** 0.5 )
        
        max_grid_size = 1.5 * max(last_layer_sizes)

        # Calculate layer points and grid sizes

        # Remove some points from curve
        tck, u_aux = splprep([curve[:, 0], curve[:, 1]], s=0)
        u = np.linspace(0, 1, num=50)
        tulple_points = splev(u, tck)
        curve = np.zeros((u.shape[0], 2))
        curve[:, 0], curve[:, 1] = np.asarray(list(tulple_points[0])), np.asarray(list(tulple_points[1]))

        # Remove some points from last_layer_sizes
        f = interp1d(u_aux, last_layer_sizes, kind='quadratic')
        last_layer_sizes_aux = f(u)

        # Dissipation
        aux = np.copy(last_layer_sizes_aux)
        for _ in range(10):
            for i in range(last_layer_sizes_aux.shape[0]):
                if i != 0 and i != last_layer_sizes_aux.shape[0] - 1:
                    aux[i] = (last_layer_sizes_aux[i - 1] + 2 * last_layer_sizes_aux[i] + last_layer_sizes_aux[i + 1]) / 4
                elif i == 0:
                    aux[i] = (last_layer_sizes_aux[last_layer_sizes_aux.shape[0] - 1] + 2 * last_layer_sizes_aux[i] + last_layer_sizes_aux[i + 1]) / 4
                elif i == last_layer_sizes_aux.shape[0] - 1:
                    aux[i] = (last_layer_sizes_aux[i - 1] + 2 * last_layer_sizes_aux[i] + last_layer_sizes_aux[0]) / 4
            last_layer_sizes_aux = np.copy(aux)

        sizes = np.zeros((ref_layers, curve.shape[0]))
        sizes_X = np.zeros((ref_layers, curve.shape[0]))
        sizes_Y = np.zeros((ref_layers, curve.shape[0]))

        layer = np.copy(curve)
        for i in range(ref_layers):
            layer = create_layer(layer, dist[i] if i == 0 else dist[i] - dist[i - 1])
            sizes_X[i, :] = layer[:, 0]
            sizes_Y[i, :] = layer[:, 1]
            for j in range(curve.shape[0]):
                cell_size = last_layer_sizes_aux[j] * ratio[i]
                if i == ref_layers - 1:
                    sizes[i, j] = 2 * max_grid_size
                else:
                    sizes[i, j] = cell_size if cell_size < max_grid_size else max_grid_size

        return [sizes_X, sizes_Y, sizes, last_layer_sizes]

    def _gen_mesh(self, X: np.ndarray,
                        Y: np.ndarray,
                        sizes_X: np.ndarray,
                        sizes_Y: np.ndarray,
                        sizes: np.ndarray,
                        last_layer_sizes: np.ndarray,
                        out: str) -> None:
        
        gmsh.initialize()

        gmsh.model.add('airfoil')

        # Boundary layer points
        bl_points = np.zeros(X.shape, dtype=np.int64)
        for i in range(X.shape[0]):
            for j in range(Y.shape[1]):
                bl_points[i, j] = gmsh.model.geo.add_point(X[i, j], Y[i, j], 0, last_layer_sizes[j])

        # Refinement points
        ref_points = np.zeros(sizes.shape, dtype=np.int64)
        for i in range(sizes.shape[0]):
            for j in range(sizes.shape[1]):
                ref_points[i, j] = gmsh.model.geo.add_point(sizes_X[i, j], sizes_Y[i, j], 0, sizes[i, j])
        
        # Exterior points
        n = 100
        ext_points = np.zeros(n, dtype=np.int64)
        angle = np.linspace(0, 2 * np.pi, num=n + 1)

        x = self._ext_radius * np.cos(angle)
        y = self._ext_radius * np.sin(angle)

        for i in range(n):
            ext_points[i] = gmsh.model.geo.add_point(x[i], y[i], 0, self._ext_cell_size)

        # Surface and boundary layer curves
        surf_curves = np.zeros(X.shape[1], dtype=np.int64)
        bl_curves = np.zeros(X.shape[1], dtype=np.int64)
        bl_top_curves = np.zeros(X.shape[1], dtype=np.int64)

        for i in range(X.shape[1]):
            bl_curves[i] = gmsh.model.geo.add_polyline(bl_points[:, i].tolist())
            
            if i == X.shape[1] - 1:
                surf_curves[i] = gmsh.model.geo.add_line(bl_points[0, i], bl_points[0, 0])
                bl_top_curves[i] = gmsh.model.geo.add_line(bl_points[X.shape[0] - 1, i], bl_points[X.shape[0] - 1, 0])
            else:
                surf_curves[i] = gmsh.model.geo.add_line(bl_points[0, i], bl_points[0, i + 1])
                bl_top_curves[i] = gmsh.model.geo.add_line(bl_points[X.shape[0] - 1, i], bl_points[X.shape[0] - 1, i + 1])

            # Set transfinite
            gmsh.model.geo.mesh.setTransfiniteCurve(bl_curves[i], self._nf, meshType="Progression", coef=self._exp3)
            gmsh.model.geo.mesh.setTransfiniteCurve(surf_curves[i], 2)
            gmsh.model.geo.mesh.setTransfiniteCurve(bl_top_curves[i], 2)

        # Refinement curves
        ref_curves = []
        for i in range(ref_points.shape[0]):
            ref_curves.append([gmsh.model.geo.add_line(ref_points[i, j], ref_points[i, j + 1]) for j in range(ref_points.shape[1] - 1)] + [gmsh.model.geo.add_line(ref_points[i, -1], ref_points[i, 0])])
            # ref_curves.append(gmsh.model.geo.add_polyline(ref_points[i, :].tolist() + [ref_points[i, 0]]))

        # Exterior curve
        ext_curve = gmsh.model.geo.add_polyline(ext_points.tolist() + [ext_points[0]])

        # Boundary layer loops and surfaces
        bl_loops = np.zeros(X.shape[1], dtype=np.int64)
        bl_surfaces = np.zeros(X.shape[1], dtype=np.int64)

        for i in range(X.shape[1]):
            if i == X.shape[1] - 1:
                bl_loops[i] = gmsh.model.geo.add_curve_loop([surf_curves[i], bl_curves[0], -bl_top_curves[i], -bl_curves[i]])
                bl_surfaces[i] = gmsh.model.geo.add_plane_surface([bl_loops[i]])
                gmsh.model.geo.mesh.setTransfiniteSurface(bl_surfaces[i])
                gmsh.model.geo.mesh.setRecombine(2, bl_surfaces[i])
            else:
                bl_loops[i] = gmsh.model.geo.add_curve_loop([surf_curves[i], bl_curves[i + 1], -bl_top_curves[i], -bl_curves[i]])
                bl_surfaces[i] = gmsh.model.geo.add_plane_surface([bl_loops[i]])
                gmsh.model.geo.mesh.setTransfiniteSurface(bl_surfaces[i])
                gmsh.model.geo.mesh.setRecombine(2, bl_surfaces[i])
        
        bl_top_loop = gmsh.model.geo.add_curve_loop(bl_top_curves.tolist())

        # Refinement curves and surfaces
        ref_loops = []
        ref_surfaces = []
        for i in range(ref_points.shape[0]):
            ref_loops.append(gmsh.model.geo.add_curve_loop(ref_curves[i]))

        for i in range(ref_points.shape[0]):
            if i == 0:
                ref_surfaces.append(gmsh.model.geo.add_plane_surface([ref_loops[0], bl_top_loop]))
            else:
                ref_surfaces.append(gmsh.model.geo.add_plane_surface([ref_loops[i], ref_loops[i - 1]]))

        # External loops and surface
        ext_loop = gmsh.model.geo.add_curve_loop([ext_curve])
        ext_surface = gmsh.model.geo.add_plane_surface([ext_loop, ref_loops[-1]])

        # Extrude mesh
        extrude = []

        for surface in bl_surfaces:
            extrude.append(gmsh.model.geo.extrude([(2, surface)], 0, 0, 0.1, [1], [1], recombine=True))
        
        for surface in ref_surfaces:
            extrude.append(gmsh.model.geo.extrude([(2, surface)], 0, 0, 0.1, [1], [1], recombine=True))
        
        extrude.append(gmsh.model.geo.extrude([(2, ext_surface)], 0, 0, 0.1, [1], [1], recombine=True))

        # Generate
        gmsh.option.setNumber('General.Verbosity', 1)
        
        gmsh.option.setNumber("Mesh.Smoothing", 100)
        gmsh.option.setNumber("Mesh.Algorithm", 6)

        gmsh.option.setNumber("General.ExpertMode", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)

        # Physical group
        foilPhysicalGroup = gmsh.model.addPhysicalGroup(2, [extrude[i][2][1] for i in range(len(extrude) - 1)])
        externalPhysicalGroup = gmsh.model.addPhysicalGroup(2, [extrude[len(extrude) - 1][2][1]]) # [ext[4][1] for ext in extrude]
        lateralsPhysicalGroup = gmsh.model.addPhysicalGroup(2, [ext_surface] + [bl for bl in bl_surfaces] + [ref for ref in ref_surfaces] + [ext[0][1] for ext in extrude])
        domainPhysicalGroup = gmsh.model.addPhysicalGroup(3, [ext[1][1] for ext in extrude])
        gmsh.model.setPhysicalName(2, foilPhysicalGroup, "foil")
        gmsh.model.setPhysicalName(2, externalPhysicalGroup, "external")
        gmsh.model.setPhysicalName(2, lateralsPhysicalGroup, "laterals")
        gmsh.model.setPhysicalName(3, domainPhysicalGroup, "volume")

        # if '-nopopup' not in sys.argv:
        #     gmsh.fltk.run()

        gmsh.write(out)

        gmsh.finalize()

        return

if __name__ == '__main__':

    # Foils path
    path = './data/foils/'
    files = listdir(path)

    # Save path
    save_to = './data/meshes/'
    files_done = listdir(save_to)
    
    # Parameters
    ns = 700
    nt = 10
    nf = 54
    epsilon1 = 1.02
    epsilon2 = 1.02
    epsilon3 = 1.1
    delta = 0.03501813261865072
    radius = 25
    ext_grid_size = 2

    # Callable
    makefoil = foil_mesh(ns, nt, nf, delta, epsilon1, epsilon2, epsilon3, radius, ext_grid_size)

    # Create meshes
    for file in tqdm(files):
        if file != 'correct.py':
            mesh_file = file.replace('.txt', '.msh')
            if mesh_file not in files_done:
                makefoil.create(path + file, save_to + mesh_file)