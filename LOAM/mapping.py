import time

import numpy as np
import mrob
import open3d as o3d

from optimizer import LOAMOptimizer
from utils import get_pcd_from_numpy

'''
地图管理,基于
t-1时刻的 T，less_sharp & less_flap
构建全局地图
'''

class Mapper:
    COVARIANCE_CNT = 5  # 用于搜索KD-tree周边点最近的点数量, surrounding points(S')
    EDGE_VOXEL_SIZE = 0.2  # 为了使得地图点分布均匀，所采用的voxel zhang2017=0.05, A-loam=0.4
    SURFACE_VOXEL_SIZE = 0.4  # zhang2017 0.1, A-loam=0.8
    # 局部地图截断使用senser周围 500m^3 似乎并没有体现出来
    def __init__(self):
        self.init = False
        self.corners = None
        self.surfaces = None
        self.position = np.eye(4)
        self.aligned_pcds = []  # world pcd
        self.all_edges = None  # world edge
        self.all_surfaces = None  # world surface
        self.cnt = 0  # total count for mapping

    def undistort(self, pcd, T):
        N_GROUPS = 30
        global_T = mrob.geometry.SE3(T).Ln()
        groups = np.array_split(pcd, N_GROUPS)
        pcds_cut = []
        for i in range(N_GROUPS):
            pcds_cut.append(mrob.geometry.SE3((i + 1) / N_GROUPS * global_T).transform_array(groups[i][:, :3]))

        reformed = np.vstack(pcds_cut)
        return reformed
    '''
    pcd[xyzr](t)
    T为每轮配准结果(t-1)
    其余为用来建图的特征点（less）(t-1)
    '''
    def append_undistorted(self, pcd, T, edge_points, surface_points, vis=False):

        self.cnt += 1
        if not self.init:  # 初始化第一帧，直接添加为o3d.pcd
            self.init = True
            pcd = get_pcd_from_numpy(pcd)
            pcd.paint_uniform_color([0, 0, 1])  # blue
            self.aligned_pcds.append(pcd)  # 全部点云
            self.all_edges = get_pcd_from_numpy(np.vstack(edge_points))
            self.all_surfaces = get_pcd_from_numpy(np.vstack(surface_points))
        else:
            prior_position = T @ self.position  # 更新position
            # 降采样
            edge_points = np.asarray(self.filter_pcd(get_pcd_from_numpy(np.vstack(edge_points)), 'edge').points)
            surface_points = np.asarray(self.filter_pcd(get_pcd_from_numpy(np.vstack(surface_points)), 'surface').points)
            # 投影cFrame到world coord
            # restored_pcd = np.vstack(pcd)
            transformed_pcd = mrob.geometry.SE3(prior_position).transform_array(np.vstack(pcd))  # pcd[t]@T
            transformed_edge_points = mrob.geometry.SE3(prior_position).transform_array(np.vstack(edge_points))  # edge@T
            transformed_surface_points = mrob.geometry.SE3(prior_position).transform_array(np.vstack(surface_points))  # surface@T
            # world构建edge-KDtree surface-KDtree
            edges_kdtree = o3d.geometry.KDTreeFlann(self.all_edges)
            surfaces_kdtree = o3d.geometry.KDTreeFlann(self.all_surfaces)
            '''
            基于cFrame的点 寻找KD-tree(world)中满足条件的最近COVARIANCE_CNT=5个点
            所有点到当前特征点的几何距离都不大于1
            '''
            edges = []
            edge_A = []
            edge_B = []
            for ind in range(len(edge_points)):
                point = transformed_edge_points[ind]
                #  can be replaced with RKNN=search_hybrid_vector_3d()，速度慢1-3倍
                # _, idx1, dists1 = edges_kdtree.search_hybrid_vector_3d(point, radius=1, max_nn=self.COVARIANCE_CNT)
                _, idx, dists = edges_kdtree.search_knn_vector_3d(point, self.COVARIANCE_CNT)  # 最近5个点
                # max(sqrt(x^2+y^2+z^2))<1 //zhang2017只考虑0.1^3内的点，由于体素化显然不满足
                if np.max(np.linalg.norm(np.asarray(self.all_edges.points)[idx] - point, axis=1)) < 1:
                    #point1 = np.asarray(self.all_edges.points)[idx]
                    #point2 = np.asarray(self.all_edges.points)[idx1]
                    status, point_a, point_b = self.is_edge(np.asarray(self.all_edges.points)[idx])  # 对应的5个点
                    if status:
                        edges.append(point)
                        edge_A.append(point_a)
                        edge_B.append(point_b)

            surfaces = []
            surface_A = []
            surface_B = []
            for ind in range(len(surface_points)):
                point = transformed_surface_points[ind]
                _, idx, dists = surfaces_kdtree.search_knn_vector_3d(point, self.COVARIANCE_CNT)
                if np.max(np.linalg.norm(np.asarray(self.all_surfaces.points)[idx] - point, axis=1)) < 1:
                    status, norm, norm_reversed = self.is_surface(np.asarray(self.all_surfaces.points)[idx])
                    if status:
                        surfaces.append(point)
                        surface_A.append(norm)
                        surface_B.append(norm_reversed)

            if len(edges) > 0 and len(surfaces) > 0:
                # edge=[edge, centroid_A, centroid_B]
                # planer=[planer, norm, norm_reversed]
                optimizer = LOAMOptimizer((np.vstack(edges), np.vstack(edge_A), np.vstack(edge_B)),
                                          (np.vstack(surfaces), np.vstack(surface_A), np.vstack(surface_B)))
                T = optimizer.optimize_2()
                self.position = T.T() @ prior_position
                self.aligned_pcds.append(get_pcd_from_numpy(T.transform_array(transformed_pcd)))
                if self.cnt % 3 == 0:
                    self.all_edges += get_pcd_from_numpy(T.transform_array(transformed_edge_points))
                    self.all_surfaces += get_pcd_from_numpy(T.transform_array(transformed_surface_points))
                    self.all_edges = self.filter_pcd(self.all_edges, 'edge')
                    self.all_surfaces = self.filter_pcd(self.all_surfaces, 'surface')
            else:
                self.aligned_pcds.append(get_pcd_from_numpy(transformed_pcd))
                self.all_edges += get_pcd_from_numpy(transformed_edge_points)
                self.all_surfaces += get_pcd_from_numpy(transformed_surface_points)

            if vis:
                o3d.visualization.draw_geometries(self.aligned_pcds)
    '''
    edge(world)到当前特征最近的COVARIANCE_CNT=5个点,构成S‘
    求S'的中心点Centroid
    求S'的协方差矩阵M，以及对应的特征值V和特征向量E
    当满足 一大两小 时，当前的S’能够形成边/线，其中最大特征值对应的特征向量，就是线的方向向量（已经归一化了）。
    基于方向向量，分别构建正向(法向×0.1) 和 反向(法向×-0.1) 到中心点的距离=0.1的两个点作为特征点a和b
    '''
    def is_edge(self, surrounded_points):
        assert surrounded_points.shape[0] == self.COVARIANCE_CNT
        centroid = np.sum(surrounded_points, axis=0) / self.COVARIANCE_CNT  # 5个点的中心点
        '''
        求参考点的协方差矩阵M，以及特征值 V 和 特征向量 E
        V contains one eigenvalue significantly larger than the other two,
        the eigenvector in E associated with the largest eigenvalue represents the orientation of the edge line
        '''
        covariance_mat = np.zeros((3, 3))
        for i in range(self.COVARIANCE_CNT):
            diff = (surrounded_points[i] - centroid).reshape((3, 1))
            covariance_mat += diff @ diff.T

        v, e = np.linalg.eig(covariance_mat)  # 返回协方差的特征值v（无序），特征向量e（归一化）
        sorted_v_ind = np.argsort(v)  # 特征值排序 由小到大
        sorted_v = v[sorted_v_ind]
        sorted_e = e[sorted_v_ind]
        unit_direction = sorted_e[:, 2]  # 特征值最大的特征向量, 代表边特征线的方向，已经归一化了
        # 特征有效性判断
        if sorted_v[2] > 3 * sorted_v[1]:  # 最大的特征值 > 3*第二大的特征值（significantly larger）
            point_a = 0.1 * unit_direction + centroid  # 正向点
            point_b = -0.1 * unit_direction + centroid  # 反向点
            return True, point_a, point_b
        else:
            return False, None, None
    '''
    并没有使用zhang2017中V contains two large eigenvalues with the third one significantly smaller的条件来判断平面。
    surface(world)到当前特征最近的COVARIANCE_CNT=5个点,构成S‘
    基于S'构造A0,B0=[-1,-1,...,-1],基于最小二乘求A0@x=B0中的x=norm
    将A0点带入求解结果norm和norm_reversed中，判断平面是否平整
    满足条件时，输出：norm和norm_reversed
    这种条件下，点到平面距离公式计算更简单
    D=|norm @ point+d|/|norm|=|ax+by+cz+d|/sqrt(a^2+b^2+c^2)
    考虑norm进行了归一化，|norm|=1可以省略
    '''
    def is_surface(self, surrounded_points):
        mat_A0 = surrounded_points
        mat_B0 = -np.ones((self.COVARIANCE_CNT, ))

        #norm = np.linalg.lstsq(mat_A0, mat_B0)[0]
        norm = np.linalg.lstsq(mat_A0, mat_B0, rcond=None)[0]  # 最小二乘求 A0 @ x = B0, norm与S'构建的平面平行

        norm_reversed = 1 / np.linalg.norm(norm)  #
        norm /= np.linalg.norm(norm)  # 归一化

        plane_valid = True
        for j in range(self.COVARIANCE_CNT):
            if np.abs(np.dot(norm, surrounded_points[j]) + norm_reversed) > 0.2:  # 平面内的点与法向点积的绝对值为0, >0.2意味着点不再平面内
                plane_valid = False
                break

        if plane_valid:
            return True, norm, norm_reversed
        else:
            return False, None, None

    # 基于特征预设voxel降采样
    def filter_pcd(self, pcd, type):
        if type == 'edge':
            return pcd.voxel_down_sample(self.EDGE_VOXEL_SIZE)
        elif type == 'surface':
            return pcd.voxel_down_sample(self.SURFACE_VOXEL_SIZE)
