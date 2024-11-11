import trimesh

# import xatlas
import cv2
import numpy as np
import random
import open3d as o3d

# from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.spatial import cKDTree
import math
from PIL import Image
import os

THRESHOLD = 0
POINT_CLOUD_SIZE = 0.2


def fill_edge_with_adjacent_color(img, idx):
    height, width = img.shape[:2]

    for x in range(width):
        if img[0, x][0] == 255 - idx:
            img[0, x] = img[1, x]
    for x in range(width):
        if img[height - 1, x][0] == 255 - idx:
            img[height - 1, x] = img[height - 2, x]
    for y in range(height):
        if img[y, 0][0] == 255 - idx:
            img[y, 0] = img[y, 1]
    for y in range(height):
        if img[y, width - 1][0] == 255 - idx:
            img[y, width - 1] = img[y, width - 2]

    return img


def load_obj(objFilePath):
    with open(objFilePath, "r") as file:
        vertices = []
        uvs = []
        normals = []
        faces = []
        for line in file:
            if line.startswith("v "):
                vertices.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith("vn "):
                normals.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith("vt "):
                uvs.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith("f "):
                face = line.strip().split()[1:]
                faces.append([int(f.split("/")[0]) - 1 for f in face])
    return np.array(vertices), np.array(normals), np.array(uvs), np.array(faces)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def barycentric(A, B, C, P):
    x = P[0]
    y = P[1]
    c1 = (x * (B[1] - C[1]) + (C[0] - B[0]) * y + B[0] * C[1] - C[0] * B[1]) / (
        A[0] * (B[1] - C[1]) + (C[0] - B[0]) * A[1] + B[0] * C[1] - C[0] * B[1]
    )
    c2 = (x * (C[1] - A[1]) + (A[0] - C[0]) * y + C[0] * A[1] - A[0] * C[1]) / (
        B[0] * (C[1] - A[1]) + (A[0] - C[0]) * B[1] + C[0] * A[1] - A[0] * C[1]
    )
    # c3 = (x*(A[1] - B[1]) + (B[0] - A[0])*y + A[0]*B[1] - B[0]*A[1]) / (C[0]*(A[1] - B[1]) + (B[0] - A[0])*C[1] + A[0]*B[1] - B[0]*A[1])
    c3 = 1 - c1 - c2
    return c1, c2, c3


def generate_masks(chart_img):
    masks = {}
    height, width, _ = chart_img.shape
    for i in range(height):
        for j in range(width):
            color = chart_img[i][j]
            if (color == [0, 0, 0]).all():
                continue
            chart_idx = 255 - color[0]
            if chart_idx not in masks.keys():
                masks[chart_idx] = []
            masks[chart_idx].append([i, j])

    return masks


def distance_interpolation(knn_dist, tri_colors):
    sum = knn_dist[0] + knn_dist[1] + knn_dist[2]
    c1 = 1 - knn_dist[0] / sum
    c2 = 1 - knn_dist[1] / sum
    c3 = 1 - knn_dist[2] / sum
    c_sum = c1 + c2 + c3

    return (
        c1 / c_sum * tri_colors[0]
        + c2 / c_sum * tri_colors[1]
        + c3 / c_sum * tri_colors[2]
    )


def parallel_projection_v0(
    P_coord, closest_point, closest_point_color, distance, n_vector
):
    v = closest_point - P_coord
    v_unit = v / np.linalg.norm(v)
    n_vector_unit = n_vector / np.linalg.norm(n_vector)
    cos_theta = np.dot(v_unit, n_vector_unit)
    sin_theta = math.sqrt(1 - cos_theta**2)
    proj_dist = distance * sin_theta
    if proj_dist < POINT_CLOUD_SIZE:
        return closest_point_color
    return np.array([255, 255, 255])  # white


def parallel_projection_v1(kdtree, P_coord, colors_array, n_vector):
    n_vector_unit = n_vector / np.linalg.norm(n_vector)
    closest_dist, closest_idx = kdtree.query(x=P_coord, k=1)

    step_P_coord = P_coord
    step_closest_dist = closest_dist
    cnt = 0

    # + n_vector direction
    positive_flag = True
    while step_closest_dist > POINT_CLOUD_SIZE:
        step_P_coord = step_P_coord + n_vector_unit * step_closest_dist
        next_step_closest_dist, closest_idx = kdtree.query(x=step_P_coord, k=1)
        if next_step_closest_dist > step_closest_dist:
            cnt += 1
        if cnt >= 2:
            positive_flag = False
            break
        step_closest_dist = next_step_closest_dist
    if positive_flag:
        return colors_array[closest_idx]

    step_P_coord = P_coord
    step_closest_dist = closest_dist
    cnt = 0

    # - n_vector direction
    negative_flag = True
    while step_closest_dist > POINT_CLOUD_SIZE:
        step_P_coord = step_P_coord - n_vector_unit * step_closest_dist
        next_step_closest_dist, closest_idx = kdtree.query(x=step_P_coord, k=1)
        if next_step_closest_dist > step_closest_dist:
            cnt += 1
        if cnt >= 2:
            negative_flag = False
            break
        step_closest_dist = next_step_closest_dist
    if negative_flag:
        return colors_array[closest_idx]

    return np.array([255, 255, 255])


def parallel_projection_v2(kdtree, P_coord, colors_array, n_vector):
    n_vector_unit = n_vector / np.linalg.norm(n_vector)
    top3_dists, top3_idxs = kdtree.query(x=P_coord, k=3)

    step_P_coord = P_coord
    step_closest_dist = top3_dists[0]
    positive_ans_top3_idxs = top3_idxs
    positive_ans_top3_dists = top3_dists
    cnt = 0

    # + n_vector direction
    while cnt < 2:
        step_P_coord = step_P_coord + n_vector_unit * step_closest_dist
        next_step_top3_dists, top3_idxs = kdtree.query(x=step_P_coord, k=3)
        next_step_closest_dist = next_step_top3_dists[0]
        # closest_idx = top3_idxs[0]
        if next_step_closest_dist <= step_closest_dist:
            positive_ans_top3_idxs = top3_idxs
            positive_ans_top3_dists = next_step_top3_dists
        else:
            cnt += 1
        step_closest_dist = next_step_closest_dist

    step_P_coord = P_coord
    step_closest_dist = top3_dists[0]
    negative_ans_top3_idxs = top3_idxs
    negative_ans_top3_dists = top3_dists
    cnt = 0

    # - n_vector direction
    while cnt < 2:
        step_P_coord = step_P_coord - n_vector_unit * step_closest_dist
        next_step_top3_dists, top3_idxs = kdtree.query(x=step_P_coord, k=3)
        next_step_closest_dist = next_step_top3_dists[0]
        # closest_idx = top3_idxs[0]
        if next_step_closest_dist <= step_closest_dist:
            negative_ans_top3_idxs = top3_idxs
            negative_ans_top3_dists = next_step_top3_dists
        else:
            cnt += 1
        step_closest_dist = next_step_closest_dist

    if positive_ans_top3_dists[0] < negative_ans_top3_dists[0]:
        return distance_interpolation(
            positive_ans_top3_dists, colors_array[positive_ans_top3_idxs]
        )
    else:
        return distance_interpolation(
            negative_ans_top3_dists, colors_array[negative_ans_top3_idxs]
        )


def samplePointCloudByTexture(
    chart_img, faces, uvs, vertices, point_cloud, kdtree, neighbors_for_color=1
):
    """
    纹理到点云的采样
        PARAM   chart_img: 初始纹理映射
                faces: 网格顶点索引
                uvs: 网格顶点uv坐标
                vertices: 网格顶点坐标
                point_cloud: 点云，形状为(N, 6), 格式xyzrgb
                kdtree: 由点云构造的kd树
                k_neighbors: 邻居数量
        RETURN  chart_img: 覆写后的RGB纹理
                sampled_points: 采样后的点
                sampled_uvs: 采样后的uv
                sampled_points_belong_triangles: key->采样点所在的mesh三角面对应的索引, value->采样点的坐标和uv对应的索引
    """
    height, width, _ = chart_img.shape
    sampled_points = []  # xyzrgb
    sampled_uvs = []
    sampled_points_belong_triangles = {}

    colors_array = point_cloud[:, 3:6]
    points_array = point_cloud[:, :3]

    print("sample point cloud and render RGB texture...")

    step_u = 1 / (width * 2)
    step_v = 1 / (height * 2)

    pos_uv_ind = 0

    for i in tqdm(range(height), leave=False):
        for j in tqdm(range(width), leave=False):
            if (chart_img[i][j] == [0, 0, 0]).all():
                continue

            u = j / width + step_u
            v = i / height + step_v
            P_uv = np.array([u, v])
            for i_ind, ind in enumerate(faces):
                uv_tri = uvs[ind]
                coord_tri = vertices[ind]

                lambda1, lambda2, lambda3 = barycentric(
                    uv_tri[0], uv_tri[1], uv_tri[2], P_uv
                )

                if (
                    lambda1 >= THRESHOLD
                    and lambda2 >= THRESHOLD
                    and lambda3 >= THRESHOLD
                ):
                    P_coord = (
                        lambda1 * coord_tri[0]
                        + lambda2 * coord_tri[1]
                        + lambda3 * coord_tri[2]
                    )

                    knn_dist, knn_idx = kdtree.query(x=P_coord, k=neighbors_for_color)
                    if neighbors_for_color > 1:
                        # avg_color = np.mean(colors_array[knn_idx], axis=0)
                        avg_color = distance_interpolation(
                            knn_dist, colors_array[knn_idx]
                        )
                        chart_img[i][j] = avg_color
                    else:
                        n_vector = np.cross(
                            coord_tri[1] - coord_tri[0], coord_tri[2] - coord_tri[0]
                        )
                        # chart_img[i][j] = parallel_projection_v0(P_coord, points_array[knn_idx], colors_array[knn_idx], knn_dist, n_vector)
                        chart_img[i][j] = parallel_projection_v2(
                            kdtree, P_coord, colors_array, n_vector
                        )

                    sampled_points.append(P_coord)
                    sampled_uvs.append(P_uv)

                    if i_ind not in sampled_points_belong_triangles.keys():
                        sampled_points_belong_triangles[i_ind] = []
                    sampled_points_belong_triangles[i_ind].append(pos_uv_ind)

                    pos_uv_ind += 1
                    break

    sampled_points = np.array(sampled_points)
    sampled_uvs = np.array(sampled_uvs)

    print("len", len(sampled_points))
    # np.savetxt("./sampled_points.txt", sampled_points)

    return chart_img, sampled_points, sampled_uvs, sampled_points_belong_triangles


def estimate_normal(point, neighbors):
    """
    估计点云中某一点的法线
        PARAM   point: 点云中某一点, 形状(3,)
                neighbors: 邻域点, 形状(N, 3)
        RETURN  法线向量
    """
    covariance_matrix = np.cov(neighbors - point, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    normal = eigenvectors[:, np.argmin(eigenvalues)]

    return normal


def renderTexture(chart_img, faces, uvs, vertices, point_cloud, neighbors_for_normal=5):
    """
    CPU渲染法线贴图
        PARAM   chart_img: 初始纹理映射
                indices: 网格顶点索引
                uvs: 网格顶点uv坐标
                vertices: 网格顶点坐标
                point_cloud: 点云，形状为(N, 6), 格式xyzrgb
                kdtree: 由点云构造的kd树
                k_neighbors: 邻居点的数量
        RETURN  rgb_texture: RGB贴图
                normal_texture: 法线贴图
    """
    rgb_texture = np.copy(chart_img)
    normal_texture = np.copy(chart_img)

    points_array = point_cloud[:, :3]
    colors_array = point_cloud[:, 3:6]
    kdtree = cKDTree(points_array)

    rgb_texture, sampled_points, sampled_uvs, sampled_points_belong_triangles = (
        samplePointCloudByTexture(
            rgb_texture, faces, uvs, vertices, point_cloud, kdtree
        )
    )

    height, width, _ = chart_img.shape
    displacements = np.zeros((height, width))

    print("render normal and displacement texture...")

    step_u = 1 / (width * 2)
    step_v = 1 / (height * 2)

    for i_ind in tqdm(sampled_points_belong_triangles.keys(), leave=False):
        tri_ind = faces[i_ind]
        tri_coord = vertices[tri_ind]
        tri_uv = uvs[tri_ind]

        e1 = tri_coord[1] - tri_coord[0]
        e2 = tri_coord[2] - tri_coord[0]
        delta_uv1 = tri_uv[1] - tri_uv[0]
        delta_uv2 = tri_uv[2] - tri_uv[0]

        r = 1.0 / (delta_uv1[0] * delta_uv2[1] - delta_uv1[1] * delta_uv2[0])
        tangent = (e1 * delta_uv2[1] - e2 * delta_uv1[1]) * r
        bitangent = (e2 * delta_uv1[0] - e1 * delta_uv2[0]) * r
        tangent = tangent / np.linalg.norm(tangent)
        bitangent = bitangent / np.linalg.norm(bitangent)

        t = tangent
        b = bitangent
        n = np.cross(t, b)
        tbn_matrix = np.array([t, b, n]).T

        # print(tbn_matrix)

        for pos_uv_ind in tqdm(sampled_points_belong_triangles[i_ind], leave=False):
            P_coord = sampled_points[pos_uv_ind]
            P_uv = sampled_uvs[pos_uv_ind]

            distances, knn_idx = kdtree.query(x=P_coord, k=neighbors_for_normal)
            P_normal = estimate_normal(P_coord, points_array[knn_idx])

            dir = points_array[knn_idx[0]] - P_coord

            tangent_space_normal = np.matmul(
                P_normal.reshape(1, 3), tbn_matrix
            ).reshape(
                3,
            )
            tangent_space_dir = np.matmul(dir.reshape(1, 3), tbn_matrix).reshape(
                3,
            )

            tangent_space_normal = tangent_space_normal / np.linalg.norm(
                tangent_space_normal
            )
            tangent_space_dir = tangent_space_dir / np.linalg.norm(tangent_space_dir)

            if tangent_space_normal[2] < 0:
                tangent_space_normal[2] *= -1

            j = round((P_uv[0] - step_u) * width)
            i = round((P_uv[1] - step_v) * height)

            displacements[i][j] = distances[0]
            if np.dot(tangent_space_dir, tangent_space_normal) < 0:
                displacements[i][j] *= -1

            normal_texture[i][j] = (tangent_space_normal + 1) / 2 * 255

    displacements = normalization(displacements)
    displacement_texture = np.repeat(displacements * 255, 3, axis=1).reshape(
        height, width, 3
    )

    return rgb_texture, normal_texture, displacement_texture


name = "simple_building_demo"

# We use trimesh (https://github.com/mikedh/trimesh) to load a mesh but you can use any library.
vertices, normals, uvs, faces = load_obj(
    "./buildings_with_bottom/simple_building_demo_remesh.obj"
)
# print(vertices)
# print(faces)
# print(uvs)

chart_image = np.asarray(Image.open("./charts/simple_building_demo_charts00.tga"))
print(chart_image.shape)
# print(image.format)
# print(image.size)
# print(image.mode)
# print(np.asarray(image))

point_cloud = np.loadtxt("./buildings_without_bottom/simple_building_demo.txt")
points_array = point_cloud[:, :3]
kdtree = cKDTree(points_array)


##########################################################################################

# rgb_texture, normal_texture, displacement_texture = renderTexture(chart_image, faces, uvs, vertices, point_cloud, neighbors_for_normal=5)

# cv2.imwrite(name + "_rgb_texture.png", rgb_texture[:, :, [2, 1, 0]])
# print("RGB write done")

# cv2.imwrite(name + "_normal_texture.png", normal_texture[:, :, [2, 1, 0]])
# print("normal write done")

# cv2.imwrite(name + "_displacement_texture.png", displacement_texture)
# print("displacement write done")

##########################################################################################

# rgb_texture = cv2.imread("simple_building_demo_rgb_texture.png")
# rgb_texture = rgb_texture[:, :, [2, 1, 0]]

# masks = generate_masks(chart_image)
# # print(len(masks.keys()))

# for key, value in masks.items():
#     print("key:", key)
#     value = np.array(value)
#     # print(value.shape)
#     [minI, minJ] = value.min(axis=0)
#     [maxI, maxJ] = value.max(axis=0)
#     height = maxI - minI + 1
#     width = maxJ - minJ + 1
#     sub_chart = np.zeros((height, width, 3))
#     for v in value:
#         sub_chart[v[0] - minI][v[1] - minJ] = rgb_texture[v[0]][v[1]]
#     cv2.imwrite(
#         "./sub_charts/sub_chart_" + str(key) + ".png", sub_chart[:, :, [2, 1, 0]]
#     )


##########################################################################################
# test normal estimation

# normals = []

# for point in points_array:
#     _, knn_idx = kdtree.query(x=point, k=5)
#     neighbors = points_array[knn_idx]
#     normal = estimate_normal(point, neighbors)
#     normals.append(normal)

# normals = np.array(normals)

# pcd = o3d.io.read_point_cloud("./b984_with_bottom.txt", format="xyzrgb")
# pcd.normals = o3d.utility.Vector3dVector(normals)
# o3d.visualization.draw_geometries([pcd], point_show_normal=True)http://www.yanhuangxueyuan.com/doc/Three.js/normalMap.html


##########################################################################################

for filename in os.listdir("./sub_charts/"):
    file_path = os.path.join("./sub_charts/", filename)
    idx = int(filename[-5])
    print(idx)
    if os.path.isfile(file_path):
        # print(file_path)
        sub_chart = cv2.imread(file_path)
        sub_chart = sub_chart[:, :, [2, 1, 0]]
        sub_chart = fill_edge_with_adjacent_color(sub_chart, idx)
        cv2.imwrite(file_path, sub_chart[:, :, [2, 1, 0]])
