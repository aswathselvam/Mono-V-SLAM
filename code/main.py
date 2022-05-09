from load_dataset import get_image_paths, get_poses, get_K
from compute_pose import *
from map import Map
from match import *
import cv2
from GetInliersRANSAC import *
from EssentialMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from visualization_utils import *
from DisambiguateCameraPose import *

def H_matrix(rot_mat, t):
    i = np.column_stack((rot_mat, t))
    a = np.array([0, 0, 0, 1])
    H = np.vstack((i, a))
    return H

img_paths = get_image_paths()
# gt_poses = get_poses()
P0 = get_K()
map = Map()

# print(gt_poses)
#
# for gt in gt_poses:
#     plt.scatter(gt[0][3], -gt[2][3], c='r', s=4)
# plt.show()

K, _, _, _, _, _, _ = cv2.decomposeProjectionMatrix(P0)
#visualizer = CameraPoseVisualizer([-20, 20], [-20, 20], [-20, 20])

p_0 = np.array([0, 0, 0, 1]).T
H_init = np.identity(4)
C = np.zeros((3, 1))
R = np.identity(3)
traj = []

for i in range(0,len(img_paths)-1):
    print(i)
    img1 = cv2.imread(img_paths[i],0)
    img2 = cv2.imread(img_paths[i+1],0)

    w = img1.shape[1]
    h = img1.shape[0]

    det = cv2.SIFT_create()
    kp1, des1 = det.detectAndCompute(img1,None)
    kp2, des2 = det.detectAndCompute(img2,None)

    matches = FLANN_matcher(des1,des2,2)

    filter_matches = filter_matches_distance(matches, 0.5)

    image1_points = np.float32([kp1[m.queryIdx].pt for m in filter_matches])
    image2_points = np.float32([kp2[m.trainIdx].pt for m in filter_matches])
    #
    # visualizer = CameraPoseVisualizer([-20, 20], [-20, 20], [-20, 20])

    F, _ = cv2.findFundamentalMat(np.float32(image1_points).reshape(-1,1,2),np.float32(image2_points).reshape(-1,1,2),cv2.FM_RANSAC) #GetInliersRANSAC(np.float32(image1_points), np.float32(image2_points), w, h)
    E = EssentialMatrixFromFundamentalMatrix(F, K)

    R_set, C_set = ExtractCameraPose(E)
    X_set = []

    for n in range(0, 4):
        X1 = LinearTriangulation(K, np.zeros((3, 1)), np.identity(3), C_set[n].T, R_set[n], np.float32(image1_points), np.float32(image2_points))
        X_set.append(X1)

    # colors = ['r', 'g', 'b', 'y']
    # for i in range(len(C_set)):
    #     RC = np.eye(4)
    #     RC[0:3, 0:3] = R_set[i]
    #     RC[0:3, 3] = C_set[i].reshape(-1, )
    #     visualizer.plot_points(X_set[i], colors[i])
    #     visualizer.extrinsic2pyramid(RC, colors[i], 5)
    #
    # RC = np.eye(4)
    # RC[0:3, 3] = 0
    # visualizer.extrinsic2pyramid(RC, 'k', 5)
    # visualizer.show()

    X, R, C = DisambiguateCameraPose(C_set, R_set, X_set)
    # visualizer = CameraPoseVisualizer([-20, 20], [-20, 20], [-20, 20])
    # RC = np.eye(4)
    # RC[0:3, 3] = 0
    # visualizer.extrinsic2pyramid(RC, colors[0], 5)
    # RC[0:3, 0:3] = R
    # RC[0:3, 3] = C.reshape(-1, )
    # visualizer.extrinsic2pyramid(RC, colors[1], 5)
    # visualizer.plot_points(X, colors[1])
    # visualizer.show()

    H_init = H_init @ H_matrix(R, C.reshape(3,1))
    # print(H_init)
    p_projection = H_init @ p_0

    #
    # C = H_init[0:3,3]
    # R = H_init[0:3,0:3]
    # print(C,R)

    traj.append([p_projection[0], -p_projection[2]])

traj = np.array(traj)
plt.scatter(traj[:,0], traj[:,1], c='r', s=4)
plt.show()

    # traj.append(C)

    # visualizer = CameraPoseVisualizer([-20, 20], [-20, 20], [-20, 20])
    # RC = np.eye(4)
    # RC[0:3,3] = 0
    # # visualizer.extrinsic2pyramid(RC, 'r', 5)
    # RC[0:3,0:3] = R
    # RC[0:3,3] = C.reshape(-1,)
    # visualizer.extrinsic2pyramid(RC, 'b', 5)
    # visualizer.plot_points(X, 'g')

# traj=np.array(traj)
# # fig = plt.figure()
# # ax = fig.add_subplot(projection='3d')
# # plt.ion()
# plt.scatter(traj[:,0],traj[:,0], c='r', s=4)
# plt.show()



# for pose, pointclouds in compute_camera_pose(img_paths, K):
#     print(pose)

    # Update data in the Map
    
    
    # Choose Keyframes


    # Perform BA


    # Detect and perform loop closure


    # Plot the Map