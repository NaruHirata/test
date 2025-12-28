import pyvista as pv
import numpy as np
import sys

args = sys.argv
in_mesh = args[1]
out_mesh = in_mesh + ".txt"
mesh = pv.read(in_mesh)
vertices = mesh.points
faces = mesh.faces.reshape(-1, 4)[:, 1:]+1 # 1-based
with open(out_mesh, 'a') as f:
    f.write(str(len(vertices)))
    f.write('\n')
    for i, (x, y, z) in enumerate(vertices, start=1):
        f.write(f"{i} {x} {y} {z}\n")
    f.write(str(len(faces)))
    f.write('\n')
    for i, (v0, v1, v2) in enumerate(faces, start=1):
        f.write(f"{i} {v0} {v1} {v2}\n")