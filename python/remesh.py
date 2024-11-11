import numpy as np
import open3d as o3d
from PIL import Image
import trimesh

# ---------- generate 2d mesh for CDT ----------
# origin = [0, 0]
# width = 264
# height = 418

# window_origin = [11, -28]
# window_width = 24
# window_height = 38
# seperate_width = 54
# seperate_height = 83
# repeat_h = 5
# repeat_v = 5

# total_points = total_edges = (repeat_h * repeat_v + 1) * 4

# f = open("./simple_building_demo.txt", "a")
# f.write("{} {}\n".format(total_points, total_edges))
# f.write("{} {}\n".format(origin[0], origin[1]))
# f.write("{} {}\n".format(origin[0] + width, origin[1]))
# f.write("{} {}\n".format(origin[0] + width, origin[1] - height))
# f.write("{} {}\n".format(origin[0], origin[1] - height))

# for i in range(repeat_h):
#     for j in range(repeat_v):
#         w_o_h = window_origin[0] + i * seperate_width
#         w_o_v = window_origin[1] - j * seperate_height
#         f.write("{} {}\n".format(w_o_h, w_o_v))
#         f.write("{} {}\n".format(w_o_h + window_width, w_o_v))
#         f.write("{} {}\n".format(w_o_h + window_width, w_o_v - window_height))
#         f.write("{} {}\n".format(w_o_h, w_o_v - window_height))

# for i in range(0, total_points, 4):
#     f.write("{} {}\n".format(i, i + 1))
#     f.write("{} {}\n".format(i + 1, i + 2))
#     f.write("{} {}\n".format(i + 2, i + 3))
#     f.write("{} {}\n".format(i + 3, i))


def barycentric(A, B, C, P):
    x = P[0]
    y = P[1]
    c1 = (x*(B[1] - C[1]) + (C[0] - B[0])*y + B[0]*C[1] - C[0]*B[1]) / (A[0]*(B[1] - C[1]) + (C[0] - B[0])*A[1] + B[0]*C[1] - C[0]*B[1])
    c2 = (x*(C[1] - A[1]) + (A[0] - C[0])*y + C[0]*A[1] - A[0]*C[1]) / (B[0]*(C[1] - A[1]) + (A[0] - C[0])*B[1] + C[0]*A[1] - A[0]*C[1])
    # c3 = (x*(A[1] - B[1]) + (B[0] - A[0])*y + A[0]*B[1] - B[0]*A[1]) / (C[0]*(A[1] - B[1]) + (B[0] - A[0])*C[1] + A[0]*B[1] - B[0]*A[1])
    c3 = 1 - c1 - c2
    return c1, c2, c3

def write_obj(file_path, vertices, uvs, faces):
    with open(file_path, 'w') as file:
        # for vertex in vertices:
        #     file.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')

        # for uv in uvs:
        #     file.write(f'vt {uv[0]} {uv[1]}\n')

        for i in range(len(vertices)):
            vertex = vertices[i]
            uv = uvs[i]
            file.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
            file.write(f'vt {uv[0]} {uv[1]}\n')
        
        for face in faces:
            # 1-index
            file.write(f'f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n')

def find_unique_faces(full_pattern_faces, hollow_pattern_faces):
    set1 = set(tuple(x) for x in full_pattern_faces)
    set2 = set(tuple(x) for x in hollow_pattern_faces)
    
    unique_faces = set1 - set2
    
    unique_faces = [list(x) for x in unique_faces]
    
    return unique_faces

def load_obj(objFilePath):
    with open(objFilePath, 'r') as file:
        vertices = []
        uvs = []
        normals = []
        faces = []
        for line in file:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('vn '):
                normals.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('vt '):
                uvs.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('f '):
                face = line.strip().split()[1:]
                faces.append([int(f.split('/')[0]) - 1 for f in face]) # 1-index to 0-index       
    return np.array(vertices), np.array(normals), np.array(uvs), np.array(faces)

vertices, _, uvs, faces = load_obj("./buildings_with_bottom/simple_building_demo_remesh.obj")


# print(vertices)
# print(uvs)
# print(faces)

chart_image = np.asarray(Image.open('./charts/simple_building_demo_charts00.tga'))
print(chart_image.shape)
height, width, _ = chart_image.shape

sub_chart = {}

for f in faces: 
    tri_uv = uvs[f] # (3, 2)
    center_uv = [(tri_uv[0][0] + tri_uv[1][0] + tri_uv[2][0]) / 3, (tri_uv[0][1] + tri_uv[1][1] + tri_uv[2][1]) / 3]
    # print(center_uv)
    center_pixel = chart_image[int(height * center_uv[1])][int(width * center_uv[0])]
    idx = 255 - center_pixel[0]
    if idx not in sub_chart:
        sub_chart[idx] = []
    sub_chart[idx].append(list(f))

# print(sub_chart)

chart_indices = sub_chart[2] # several faces

print(chart_indices)

indices_set = set()
for tri_indices in chart_indices:
    indices_set.add(tri_indices[0])
    indices_set.add(tri_indices[1])
    indices_set.add(tri_indices[2])

indices_set = list(indices_set)
print(indices_set)

print(vertices[indices_set])
print(uvs[indices_set])

chart_uvs = uvs[indices_set]
chart_vertices = vertices[indices_set]

clock_wise = [3, 1, 2, 0]

chart_vertices = chart_vertices[clock_wise]

print(chart_uvs[clock_wise])

cdt_full_mesh = trimesh.load('./buildings_with_bottom/cdt_full_mesh.off')
cdt_hollow_mesh = trimesh.load('./buildings_with_bottom/cdt_hollow_mesh.off')

# vertices = cdt_mesh.vertices
# print("Vertices:\n", vertices)

# faces = cdt_mesh.faces
# print("Faces:\n", faces)

tri_coords = cdt_hollow_mesh.vertices[:3, :]
print(tri_coords)

mesh_vertices = []
mesh_uvs = [[0, 0], [0.6, 0], [0.6, 1], [0, 1]]

for c_v in chart_vertices:
    mesh_vertices.append(list(c_v))

new_faces = []

# WALL
inter_coords = cdt_hollow_mesh.vertices[4:, :]
for i_c in inter_coords:
    c1, c2, c3 = barycentric(tri_coords[0], tri_coords[1], tri_coords[2], i_c)
    mesh_vertices.append(list(c1 * chart_vertices[0] + c2 * chart_vertices[1] + c3 * chart_vertices[2]))
    mesh_uvs.append(list(c1 * np.array([0, 0]) + c2 * np.array([0.6, 0]) + c3 * np.array([0.6, 1])))

mesh_faces = cdt_hollow_mesh.faces.tolist()

unique_faces = find_unique_faces(cdt_full_mesh.faces, cdt_hollow_mesh.faces)

print(unique_faces)

# WINDOW
for u_f in unique_faces:
    for v in u_f:
        mesh_vertices.append(mesh_vertices[v])
    mesh_faces.append([len(mesh_vertices)-3, len(mesh_vertices)-2, len(mesh_vertices)-1])
    if u_f[0] < u_f[2] and u_f[2] < u_f[1]:
        mesh_uvs.extend([[0.6, 0], [0.6, 1], [1, 0]])
    if u_f[1] < u_f[0] and u_f[0] < u_f[2]:
        mesh_uvs.extend([[1, 1], [1, 0], [0.6, 1]])
    if u_f[2] < u_f[1] and u_f[1] < u_f[0]:
        mesh_uvs.extend([[0.6, 1], [1, 1], [0.6, 0]])


# print(hollow_mesh_vertices)
# print(hollow_mesh_faces)

# print(len(mesh_uvs))
# print(len(mesh_vertices))

# write_obj("./inst_texture_mesh.obj", mesh_vertices, mesh_uvs, mesh_faces)