import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
from apriltag_calibrate.visualization import draw_tag, draw_3dpoints, draw_axes, draw_camera
from apriltag_calibrate.utils import KeyType
vis_camera = True
vis_master_tag = True
vis_aid_tag = False
vis_points = True
test_plane = False


if test_plane:
    vis_points = True


figure = plt.figure()
ax = figure.add_subplot(121, projection='3d')
ax_tag = figure.add_subplot(122, projection='3d')
# ax.set_box_aspect([1, 1, 1])


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to the bundle calibrate file ")

ap.add_argument("-s", "--size", required=False,
                default=0.015, help="visual size of the tag")

ap.add_argument('-test_cube', action='store_true',
                help='calculate error for cube dataset')
ap.add_argument('-test_dodecahedron', action='store_true',
                help='calculate error for dodecahedron dataset')

args = ap.parse_args()
test_cube = args.test_cube
test_dodecahedron = args.test_dodecahedron
file_path = args.input
points = []
tags = []
z_s = []
with open(file_path, 'r') as stream:
    try:
        data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
    bundle_data = data["bundle_pose"]
    tag_data = data["tag_pose"]
    camera_data = data["camera_pose"]
    extra_data = data["extra_data"]
    tag_size = extra_data["tag_size"]

    for tag, data in tag_data.items():
        chr = tag[0]
        index = int(tag[1:])
        pose = np.array(data)
        if chr == KeyType.MASTER_TAG and vis_master_tag:
            points = draw_tag(ax_tag, pose, tag_size, index)
            tags.append((pose, points))

        elif chr == KeyType.AID_TAG and vis_aid_tag:
            draw_tag(ax_tag, pose, tag_size, index)

    for bundle, data in bundle_data.items():
        chr = bundle[0]
        index = int(bundle[1:])
        pose = np.array(data)
        if chr == KeyType.BUNDLE:
            draw_axes(ax, pose, 0.1)

    for camera, data in camera_data.items():
        chr = camera[0]
        index = int(camera[1:])
        pose = np.array(data)
        if chr == KeyType.CAMERA and vis_camera:
            draw_camera(ax, pose, 0.1)

if test_plane:
    # fit plane
    points = np.array(points)

    # Calculate the mean of the points
    centroid = np.mean(points, axis=0)

    # Subtract the mean from the points
    points_centered = points - centroid

    # Perform singular value decomposition
    u, s, vh = np.linalg.svd(points_centered)

    # The normal vector of the plane is the last column in vh
    normal = vh[-1]

    # The coefficients of the plane equation are the elements of the normal vector
    # The constant term is given by the dot product of the centroid and the normal vector
    a, b, c = normal
    d = -centroid.dot(normal)

    # Calculate the distances from the points to the plane
    distances = np.abs(a * points[:, 0] + b * points[:, 1] +
                       c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)

    # Calculate the fit error as the sum of the squared distances
    fit_error = np.sum(distances**2)

    print(f"The plane fit error is {fit_error}")


if test_cube:
    figure_cube = plt.figure()
    ax_cube = figure_cube.add_subplot(111, projection='3d')

    # Split tags into six planes according to pose and points
    pose_group = [[], [], [], [], [], []]
    pose_ref = [
        np.array([0, 0, 1, 1],),
        np.array([0, 1, 0, 1],),
        np.array([1, 0, 0, 1]),
    ]
    for pose, points in tags:
        min_d = 100
        min_dd = 100
        min_index = 0

        R = pose[:3, :3]

        for i in range(3):
            ref_vec = pose_ref[i][:-1]
            pose_vec: np.ndarray = R@np.array([0, 0, 1]).T
            dd = np.dot(ref_vec, pose_vec)
            d = np.abs(1-np.abs(dd))
            if d < 0:
                pose_vec = -pose_vec
            # print(f"{i} {d}")
            if d < min_d:
                min_d = d
                min_dd = dd
                min_index = i
        if min_dd < 0:
            pose_group[min_index].append(
                (pose, points, pose_ref[min_index][:-1]))
        else:
            pose_group[min_index +
                       3].append((pose, points, pose_ref[min_index][:-1]))
    group_vec = []
    for group_id, group in enumerate(pose_group):
        avr_vec = []
        # print("_group start_______")
        for pose, points, ref_vec in group:
            R = pose[:3, :3]
            pose_vec: np.ndarray = R@np.array([0, 0, 1]).T
            d = np.dot(ref_vec, pose_vec)
            if d < 0:
                pose_vec = -pose_vec

            # print(f"vec {pose_vec.round(2)} {d:2f}")
            draw_tag(ax_cube, pose, tag_size, group_id)
            avr_vec.append(pose_vec)
        group_vec.append(np.mean(avr_vec, axis=0))
    near_angles = []
    opsite_angles = []
    for i in range(len(group_vec)):
        for j in range(i+1, len(group_vec)):
            # print(f"{i} {j} {np.dot(group_vec[i], group_vec[j])}")
            rad = np.arccos(np.dot(group_vec[i], group_vec[j]))
            if j-i == 3:
                opsite_angles.append(rad*180.0/np.pi)
            else:
                near_angles.append(rad*180.0/np.pi)
    print(f"near angles {near_angles}")
    print(f"opsite angles {opsite_angles}")
    oppsite_error = np.abs(np.array(opsite_angles)-0.0)
    near_error = np.abs(np.array(near_angles)-90.0)
    print("max error ", max(oppsite_error.max(), near_error.max()))
    print(f"avr error", np.mean(np.concatenate((oppsite_error, near_error))))


if test_dodecahedron:
    # Expected angles between outward face normals of a regular dodecahedron:
    #   63.43°  — faces sharing an edge   (dot =  1/sqrt(5))
    #  116.57°  — faces sharing a vertex  (dot = -1/sqrt(5))
    #  180.00°  — opposite faces          (dot = -1)
    ANGLE_ADJACENT = np.degrees(np.arccos(1.0 / np.sqrt(5)))   # ≈ 63.43°
    ANGLE_SKEW     = np.degrees(np.arccos(-1.0 / np.sqrt(5)))  # ≈ 116.57°
    ANGLE_OPPOSITE = 180.0
    CLUSTER_THRESHOLD = 30.0  # °, well below the 63.43° minimum inter-face angle

    figure_dodeca = plt.figure()
    ax_dodeca = figure_dodeca.add_subplot(111, projection='3d')

    # Compute outward normal (local Z) for every tag
    tag_normals = [(pose[:3, :3] @ np.array([0, 0, 1]), pose, pts)
                   for pose, pts in tags]

    # Greedy clustering: group tags whose normals are within CLUSTER_THRESHOLD
    face_groups = []
    assigned = [False] * len(tag_normals)
    for i, (ni, pi, ptsi) in enumerate(tag_normals):
        if assigned[i]:
            continue
        group = [(ni, pi, ptsi)]
        assigned[i] = True
        for j, (nj, pj, ptsj) in enumerate(tag_normals):
            if assigned[j]:
                continue
            angle = np.degrees(np.arccos(np.clip(np.dot(ni, nj), -1.0, 1.0)))
            if angle < CLUSTER_THRESHOLD:
                group.append((nj, pj, ptsj))
                assigned[j] = True
        face_groups.append(group)

    print(f"\nDodecahedron test: {len(face_groups)} face group(s) detected")

    # Mean normal per group
    group_normals = []
    for gid, group in enumerate(face_groups):
        mean_n = np.mean([n for n, _, _ in group], axis=0)
        mean_n /= np.linalg.norm(mean_n)
        group_normals.append(mean_n)
        for _, pose, _ in group:
            draw_tag(ax_dodeca, pose, tag_size, gid)

    # Classify all pairwise angles into the three expected buckets
    adjacent_angles, skew_angles, opposite_angles = [], [], []
    for i in range(len(group_normals)):
        for j in range(i + 1, len(group_normals)):
            angle = np.degrees(np.arccos(
                np.clip(np.dot(group_normals[i], group_normals[j]), -1.0, 1.0)))
            d_adj = abs(angle - ANGLE_ADJACENT)
            d_skew = abs(angle - ANGLE_SKEW)
            d_opp = abs(angle - ANGLE_OPPOSITE)
            if d_adj <= d_skew and d_adj <= d_opp:
                adjacent_angles.append(angle)
            elif d_skew <= d_opp:
                skew_angles.append(angle)
            else:
                opposite_angles.append(angle)

    all_errors = []
    if adjacent_angles:
        err = np.abs(np.array(adjacent_angles) - ANGLE_ADJACENT)
        all_errors.append(err)
        print(f"Edge-adjacent angles (expected {ANGLE_ADJACENT:.2f}°): "
              f"{[round(a,2) for a in adjacent_angles]}")
        print(f"  max error {err.max():.4f}°  avg error {err.mean():.4f}°")
    if skew_angles:
        err = np.abs(np.array(skew_angles) - ANGLE_SKEW)
        all_errors.append(err)
        print(f"Vertex-adjacent angles (expected {ANGLE_SKEW:.2f}°): "
              f"{[round(a,2) for a in skew_angles]}")
        print(f"  max error {err.max():.4f}°  avg error {err.mean():.4f}°")
    if opposite_angles:
        err = np.abs(np.array(opposite_angles) - ANGLE_OPPOSITE)
        all_errors.append(err)
        print(f"Opposite angles (expected {ANGLE_OPPOSITE:.2f}°): "
              f"{[round(a,2) for a in opposite_angles]}")
        print(f"  max error {err.max():.4f}°  avg error {err.mean():.4f}°")
    if all_errors:
        combined = np.concatenate(all_errors)
        print(f"Overall — max error {combined.max():.4f}°  "
              f"avg error {combined.mean():.4f}°")

# Set the axes to be tight to maximize the plot area
plt.axis('tight')

# Remove padding and margin
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

plt.show()
