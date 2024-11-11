import open3d as o3d
import numpy as np
import copy
import random

building = np.loadtxt("../dataset/Synthetic_v3_InstanceSegmentation/1_points_GTv3/9.txt")
building = building[:, :6] # only consider xyz and rgb

X = building[:, 0]
Y = building[:, 1]
Z = building[:, 2]

min_x = X.min()
max_x = X.max()

min_y = Y.min()
max_y = Y.max()

min_z = Z.min()

print(min_x, max_x, min_y, max_y, min_z)

bottom_plane = []

for i in range(5000):
    rx = random.random() * (max_x - min_x) + min_x
    ry = random.random() * (max_y - min_y) + min_y
    bottom_plane.append([rx, ry, min_z, 0, 0, 0])
    
bottom_plane = np.array(bottom_plane)
building = np.vstack((building, bottom_plane))

np.savetxt("./buildings_with_bottom/simple_building_demo.txt", building)