import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import random
import os


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("mkdir success")
    else:
        print("dir exists")



# 读取点云数据
# pcd = o3d.io.read_point_cloud("./buildings_with_bottom/simple_building_demo.txt", format="")  # 替换为你的点云文件路径
point_cloud = np.loadtxt("./buildings_with_bottom/simple_building_demo.txt")
points = point_cloud[:, :3]
colors = point_cloud[:, 3:]

# 设置区域生长的参数
color_threshold = 10  # 颜色差异的阈值
distance_threshold = 2  # 空间距离的阈值
min_cluster_size = 100  # 最小的分割区域点数

# 计算最近邻关系
nbrs = NearestNeighbors(n_neighbors=30, algorithm='ball_tree').fit(points)
_, indices = nbrs.kneighbors(points)

# 用于记录访问状态
visited = np.zeros(len(points), dtype=bool)
clusters = []

def region_grow(point_idx):
    """执行区域生长，提取一个符合颜色相似度和距离阈值的簇"""
    cluster = []
    queue = [point_idx]
    visited[point_idx] = True
    
    while queue:
        current_idx = queue.pop(0)
        cluster.append(current_idx)
        
        # 遍历当前点的邻域点
        for neighbor_idx in indices[current_idx]:
            if not visited[neighbor_idx]:
                # 计算颜色距离和空间距离
                color_diff = np.linalg.norm(colors[current_idx] - colors[neighbor_idx])
                spatial_dist = np.linalg.norm(points[current_idx] - points[neighbor_idx])

                # if color_diff > 0:
                #     print(color_diff, spatial_dist)
                
                if color_diff < color_threshold and spatial_dist < distance_threshold:
                    visited[neighbor_idx] = True
                    queue.append(neighbor_idx)
    
    return cluster

# 进行区域生长分割
for i in range(len(points)):
    if not visited[i]:
        cluster = region_grow(i)
        # 如果簇的大小大于最小阈值，则保留该簇
        if len(cluster) > min_cluster_size:
            clusters.append(cluster)

print(len(clusters))

remaining_indices = np.arange(len(point_cloud))

# 将每个簇转换为点云对象并显示
# for cluster in clusters:
#     cluster_pcd = o3d.geometry.PointCloud()
#     cluster_pcd.points = o3d.utility.Vector3dVector(points[cluster])
#     cluster_pcd.colors = o3d.utility.Vector3dVector(colors[cluster] / 255)
#     o3d.visualization.draw_geometries([cluster_pcd])

for cluster in clusters:
    remaining_indices = np.setdiff1d(remaining_indices, cluster)

cluster_pcd = o3d.geometry.PointCloud()
cluster_pcd.points = o3d.utility.Vector3dVector(points[remaining_indices])
cluster_pcd.colors = o3d.utility.Vector3dVector(colors[remaining_indices] / 255)
o3d.visualization.draw_geometries([cluster_pcd])


remaining_points = points[remaining_indices]

eps = 0.8  # 邻域半径
min_samples = 10  # 最小点数

db = DBSCAN(eps=eps, min_samples=min_samples).fit(remaining_points)
labels = db.labels_

num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"检测到的簇数量: {num_clusters}")

remaining_colors = np.zeros((remaining_points.shape[0], 3))  # 初始化所有点为黑色
unique_labels = set(labels)

for label in unique_labels:
    if label == -1:
        color = [0, 0, 0]
    else:
        color = [random.uniform(0, 1) for _ in range(3)]
    remaining_colors[labels == label] = color

cluster_pcd.colors = o3d.utility.Vector3dVector(remaining_colors)
o3d.visualization.draw_geometries([cluster_pcd])



# window_dic = {}

# for id in range(25):
#     print("process..." + str(id))
#     scene_block = np.loadtxt("../dataset/Synthetic_v3_InstanceSegmentation/" + str(id + 1) + "_points_GTv3.txt", delimiter=',')
#     for i in range(len(scene_block)):
#         el = scene_block[i]
#         if int(el[-2]) == 17:
#             if id not in window_dic:
#                 window_dic[id] = []
#             window_dic[id].append(el)

# print(window_dic)

# # print(building_dic)
# for key in window_dic.keys():
#     print(key, len(window_dic[key]))
#     np.savetxt("../dataset/Synthetic_v3_InstanceSegmentation/" + str(key) + "_window.txt", window_dic[key])
