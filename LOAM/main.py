import copy
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
# from loader_vlp16 import LoaderVLP16
from loader_kitti import LoaderKITTI
from mapping import Mapper
from odometry_estimator import OdometryEstimator
import utils

def find_transformation(source, target, trans_init):
    threshold = 0.2
    if not source.has_normals():
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=50))
    if not target.has_normals():
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=50))
    transformation = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,
                                                       o3d.pipelines.registration.TransformationEstimationPointToPlane()).transformation
    return transformation

def get_pcd_from_numpy(pcd_np):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
    return pcd

# 0-63, 由下至上的扫描线
def show_ring_num(pcd):
    points = pcd[0]  #
    ring = points[:, 3] / max(points[:, 3])  # unify intensity
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(plt.get_cmap('jet')(ring)[:, 0:3])  # RGBA2RGB
    utils.pc_show([pcd])


if __name__ == '__main__':
    # folder = '../../alignment/numpy/'
    folder = '/media/kid/data/ov/sequences/'  # 硬盘没有挂常驻，需要先打开硬盘
    loader = LoaderKITTI(folder, '04')  # data loader init and config the seq #, 04 contains 270 frames for short test

    odometry = OdometryEstimator()  # init odometry
    # global_transform = np.eye(4)  # global RT matrix
    pcds = []  # frames
    mapper = Mapper()  # init map management, 因为mapper自己维护了一个内部变量保存全局RT更新，上面的global_transform是没有用的。
    #'''
    #load
    for i in range(loader.length()):  # per frame iter
        if i >= 50:  # 起始帧，跳过n帧
            pcd = loader.get_item(i)  # load frame, output = 4*n matrix, [begin_idx], [end_idx]
            #aaa=pcd[0][:, 3]
            # show_ring_num(pcd)  # pcd=[xyzr] 可视化使用
            T, sharp_points, flat_points = odometry.append_pcd(pcd)  # get T and feature points @ t-1
            '''
            The mapping algorithm runs at a lower frequency than the odometry algorithm
            这里频率相同，除了可以改变可视化频率
            lego-loam建图1Hz
            '''
            #mapper.append_undistorted(pcd[0], T, sharp_points, flat_points, vis=(i % 1 == 0))  # mapping
            mapper.append_undistorted(pcd[0], T, sharp_points, flat_points, vis=(i % 10 == 0))  # mapping

    '''
    # Visual comparison with point-to-plane ICP
    pcds = []
    global_transform = np.eye(4)
    for i in range(50, 56):
        print(i)
        pcd_np_1 = get_pcd_from_numpy(loader.get_item(i)[0])
        pcd_np_2 = get_pcd_from_numpy(loader.get_item(i + 1)[0])

        T = find_transformation(pcd_np_2, pcd_np_1, np.eye(4))
        print(T)
        print(T)
        global_transform = T @ global_transform
        pcds.append(copy.deepcopy(pcd_np_2).transform(global_transform))
        pcds.append(copy.deepcopy(pcd_np_1))

    o3d.visualization.draw_geometries(pcds)
    '''