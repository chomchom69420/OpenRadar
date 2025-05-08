import csv
import glob
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import Counter
import matplotlib.colors as mcolors
import open3d as o3d
from scipy.spatial import KDTree

makeMovie = False

vicon_files = glob.glob('vicon/*.csv')
M = np.eye(4)
last_previous_refined_centroids=None #abhi tak jo bhi latest points hai apne paas

def estimate_R_with_isotropic_errors(ori_norm_arr, rotated_quats_arr):
    N = np.dot(ori_norm_arr.T, rotated_quats_arr)
    U, s, Vt = np.linalg.svd(N)
    V = Vt.T
    return np.dot(np.dot(V, np.diag([1, np.linalg.det(V @ U)])), U.T)

def estimate_transformation(R, c_x, c_y):
    T = np.eye(3)  
    T[:2, :2] = R 
    # T[:2, 2] = [c_x, 0]
    return T

def process_vicon(filename):
    vicon_df = pd.read_csv(filename, skiprows=4)
    df_coordinates = vicon_df.iloc[:, 2:]
    num_points = df_coordinates.shape[1] // 3
    new_columns = [f"Point_{i}_{coord}" for i in range(num_points) for coord in ["X", "Y", "Z"]]
    df_coordinates.columns = new_columns
    max_points = num_points
    df_filled = pd.DataFrame(columns=[f"Point_{i}_{coord}" for i in range(max_points) for coord in ["X", "Y", "Z"]])
    for index, row in df_coordinates.iterrows():
        valid_points = row.dropna().values.reshape(-1, 3)
        filled_row = pd.Series(index=df_filled.columns, dtype=float)
        for i, point in enumerate(valid_points):
            filled_row[f"Point_{i}_X"], filled_row[f"Point_{i}_Y"], filled_row[f"Point_{i}_Z"] = point
        df_filled.loc[index] = filled_row
    return df_filled

df_processed = process_vicon(vicon_files[0])

if makeMovie:
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter([], [], c='r')
    ax.set_xlim(-2500, 2500)  # Adjust limits based on your data
    ax.set_ylim(0, 6000)
    ax.grid(True)

reference_points = np.array([[-115,70],[-25,25], [80,65], [60,-160],[-135,-160]])
num_frames = len(df_processed)

def find_nearest_neighbors(source_pc, target_pc, nearest_neigh_num):
    # Find the closest neighbor for each anchor point through KDTree
    point_cloud_tree = o3d.geometry.KDTreeFlann(source_pc)
    # Find nearest target_point neighbor index
    points_arr = []
    for point in target_pc.points:
        [_, idx, _] = point_cloud_tree.search_knn_vector_3d(point, nearest_neigh_num)
        points_arr.append(source_pc.points[idx[0]])
    return np.asarray(points_arr)


def find_transform(reference_points, target_points):
    reference = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    reference.points = o3d.utility.Vector3dVector(reference_points)
    target.points = o3d.utility.Vector3dVector(target_points)
    new_target = find_nearest_neighbors(target, reference, 1)
    new_target_centroid = np.mean(new_target, axis=0)
    reference_points_centroid = np.mean(reference_points, axis=0)
    new_target_repos = np.zeros_like(new_target)
    reference_points_repos = np.zeros_like(reference_points)
    new_target_repos = np.asarray([new_target[ind] - new_target_centroid for ind in range(len(new_target))])
    reference_points_repos = np.asarray([reference_points[ind] - reference_points_centroid for ind in range(len(reference_points))])
    
    cov_mat = reference_points_repos.transpose() @ new_target_repos

    U, X, Vt = np.linalg.svd(cov_mat)
    R = U @ Vt
    t = reference_points_centroid - R @ new_target_centroid
    t = np.reshape(t, (1,3))
    curr_cost = np.linalg.norm(reference_points_repos - (R @ new_target_repos.T).T)
    transform_matrix = np.hstack((R, t.T))
    transform_matrix = np.vstack((transform_matrix, np.array([0, 0, 0, 1])))
    return transform_matrix


# Store transformed points for animation
transformed_frames = []

def find_bot_refined_centroids(points, min_samples=8, eps=500):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    label_counts = Counter(labels)
    most_common_label = max(label_counts, key=lambda x: label_counts[x] if x != -1 else 0)
    
    selected_points = points[labels == most_common_label]
    bot_centroid = np.mean(selected_points, axis=0)
    # print("bot_centroid", bot_centroid)
    fine_selected_points_clustering = DBSCAN(eps=10, min_samples=3).fit(selected_points)
    fine_labels = fine_selected_points_clustering.labels_
    refined_label_counts = Counter(fine_labels)
    
    refined_centroids = []
    for label in refined_label_counts:
        if label != -1:  
            cluster_points = selected_points[fine_labels == label]
            refined_centroids.append(np.mean(cluster_points, axis=0))
    
    refined_centroids = np.array(refined_centroids)   
    refined_centroids = refined_centroids - bot_centroid
    refined_centroids_3d = np.hstack((refined_centroids[:, :2], np.zeros((refined_centroids.shape[0], 1)))) 
    
    return refined_centroids_3d, bot_centroid, selected_points, labels

def check_bot(selected_points):
    bot_centroid = np.mean(selected_points, axis=0)
    # print("bot_centroid", bot_centroid)
    fine_selected_points_clustering = DBSCAN(eps=10, min_samples=3).fit(selected_points)
    fine_labels = fine_selected_points_clustering.labels_
    refined_label_counts = Counter(fine_labels)
    
    refined_centroids = []
    for label in refined_label_counts:
        if label != -1:  
            cluster_points = selected_points[fine_labels == label]
            refined_centroids.append(np.mean(cluster_points, axis=0))
    
    refined_centroids = np.array(refined_centroids)
    return len(refined_centroids) > 2
 
# Give the points with the bot points removed
def remove_bot_points(points, labels):
    label_counts = Counter(labels)
    most_common_label = max(label_counts, key=lambda x: label_counts[x] if x != -1 else 0)
    selected_points = points[labels == most_common_label]
    not_bot_mask = np.array([
        not np.any(np.all(pt == selected_points, axis=1))
        for pt in points
    ])
    masked_points = points[not_bot_mask]
    masked_labels = labels[not_bot_mask]
    return masked_points, masked_labels
        
def classify_points(curr_obj_points, prev_obj_points, threshold=3.0):
    """
    Classifies points as static or dynamic based on displacement between frames.

    Parameters:
        curr_obj_points (list of lists): Current frame points [[x1, y1, z1], [x2, y2, z2], ...]
        prev_obj_points (list of lists): Previous frame points [[x1, y1, z1], [x2, y2, z2], ...]
        threshold (float): Distance threshold to determine static vs. dynamic.

    Returns:
        static_points (numpy array): Points with minimal displacement.
        dynamic_points (numpy array): Points with significant displacement.
    """

    curr_obj_points = np.array(curr_obj_points)
    prev_obj_points = np.array(prev_obj_points)

    # Build KDTree for nearest neighbor search
    tree = KDTree(prev_obj_points)

    # Find nearest neighbors in previous frame
    distances, indices = tree.query(curr_obj_points)

    # Classify points based on displacement
    static_mask = distances < threshold
    static_points = curr_obj_points[static_mask]
    dynamic_points = curr_obj_points[~static_mask]

    return static_points, dynamic_points

def classfiy_static_dynamic(curr_points, prev_points, curr_labels, prev_labels):
    curr_obj_points, curr_obj_labels = remove_bot_points(curr_points, curr_labels)
    prev_obj_points, prev_obj_labels = remove_bot_points(prev_points, prev_labels)
    
    return classify_points(curr_obj_points, prev_obj_points)

def finding_bot(frame_no, df):
    global M 
    global last_previous_refined_centroids
    frame_window = 10
    eps = 500
    min_samples = 8
    start = frame_no * frame_window
    prev_start = (frame_no-1) * frame_window
    static_points = np.array([[]])
    dynamic_points = np.array([[]])
    transformed_static_points = np.array([[]])
    transformed_dynamic_points = np.array([[]])
    if start >= num_frames:
        return np.array([]), np.array([]), np.array([])   # Stop animation if frame exceeds limit
    
    frame_subset = df.iloc[start:start + frame_window].dropna(axis=1, how='all')
    points = frame_subset.values.reshape(-1, 3)
    points = points[~np.isnan(points).any(axis=1)] 
    
    if len(points) == 0:
        return np.array([])  # Skip empty frames
    
    refined_centroids_3d, bot_centroid, bot_points, curr_labels = find_bot_refined_centroids(points)

    # If M != eye(4) => NOT the first frame, store the previous refined centroids 
    if np.sum(M==np.eye(4))!=16:
        prev_frame_subset = df.iloc[prev_start:prev_start + frame_window].dropna(axis=1, how='all')
        prev_points = prev_frame_subset.values.reshape(-1, 3)
        prev_points = prev_points[~np.isnan(prev_points).any(axis=1)]
        prev_refined_centroids_3d, _, _, prev_labels = find_bot_refined_centroids(prev_points)
        if len(prev_refined_centroids_3d)==5:
            last_previous_refined_centroids=prev_refined_centroids_3d
        
        if len(refined_centroids_3d)==5 and len(prev_refined_centroids_3d)!=5:
            prev_refined_centroids_3d=last_previous_refined_centroids
            
        # Need to compare the points with the previous points to classify static, dynamic
        static_points, dynamic_points = classfiy_static_dynamic(points, prev_points, curr_labels, prev_labels)
    
    #If it is the first frame
    else:
        static_points = points
        
    # np.sum(M==np.eye(4))!=16 --> M != np.eye(4) (M has changed : not the first frame)
    if len(refined_centroids_3d)!=5 or (np.sum(M==np.eye(4))!=16 and len(prev_refined_centroids_3d) != 5):
        # print(f"Skipping frame no {frame_no}")
        return np.array([]), np.array([]), np.array([])   # Skip frames without exactly 5 refined centroids

    reference_points_3d =  np.hstack((reference_points[:, :2], np.zeros((reference_points.shape[0], 1))))
    
    if np.sum(M==np.eye(4))==16:
        transform_matrix = find_transform(reference_points_3d, refined_centroids_3d)
    else:
        transform_matrix = find_transform(prev_refined_centroids_3d, refined_centroids_3d)
    M = transform_matrix @ M
    rot_matrix = M[:3,:3]
    originated_points = points-bot_centroid
    originated_bot_points = bot_points - bot_centroid
    transformed_points = (rot_matrix @ originated_points.T).T
    transformed_bot_points = (rot_matrix @ originated_bot_points.T).T
    bot_mask = np.array([
        not np.any(np.all(pt == transformed_bot_points, axis=1))
        for pt in transformed_points
    ])
    transformed_points = transformed_points[bot_mask]  # removing points corresponding to the bot
    transformed_points = transformed_points[transformed_points[:,1]>=0]
    if np.sum(M==np.eye(4))!=16:
        originated_static_points = static_points - bot_centroid
        transformed_static_points =  (rot_matrix @ originated_static_points.T).T
        transformed_static_points = transformed_static_points[transformed_static_points[:,1]>=0]
        if dynamic_points.shape[1]==3:
            originated_dynamic_points = dynamic_points - bot_centroid
            transformed_dynamic_points =  (rot_matrix @ originated_dynamic_points.T).T
            transformed_dynamic_points = transformed_dynamic_points[transformed_dynamic_points[:,1]>=0]
        else:
            return transformed_points[:, :2], transformed_static_points[:, :2], np.array([])

    return transformed_points[:, :2], transformed_static_points[:, :2], transformed_dynamic_points[:, :2]

def update(frame_no):
    transformed_points, _, _ = finding_bot(frame_no, df_processed)
    if len(transformed_points) == 0:
        return sc,  # Skip updating scatter plot
    
    sc.set_offsets(transformed_points)
    ax.set_title(f'Frame No: {frame_no}')
    return sc,

def export_points_framewise(filename, frame_range):
    pcd = []                        #list of all pointclouds indexed by frame
    filename = f"vicon/{filename}"
    # print(filename)
    df = process_vicon(filename)
    # print(f"{len(df), df.tail()}")
    for frame_num in frame_range:
        points, _, _ = finding_bot(frame_num, df)
        pcd.append(points)
        # pcd.append(finding_bot(frame_num, df))
    
    return pcd

# Create animation
if makeMovie:
    print("Total number of frames",num_frames - 400)
    frame_range = range(60, 500)
    ani = animation.FuncAnimation(fig, update, frames=frame_range, interval=200, blit=True)
    ani.save('animation.gif', writer='pillow', fps=200)
    exit(1)
# plt.show()


    
