import pyvista as pv
import numpy as np
import sys

args = sys.argv
in_mesh = args[1]
out_center = in_mesh + ".center.txt"
mesh = pv.read(in_mesh)
centers = mesh.cell_centers()
with open(out_center, 'a') as f:
    f.write(str(len(centers.points)))
    f.write('\n')
    np.savetxt(f, centers.points)