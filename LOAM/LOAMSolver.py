import mrob
import numpy as np
import sys
#import time
#import transforms3d

# Optimization solver for LOAM task
# * Uses pre-calculated Jacobians for system of equations based on SE(3) Lie algebra
# * Uses M-estimator based on Truncated Least Squares to deal with outlier
class LOAMSolver:
    def __init__(self, use_estimators=False, region_rate=0.7):
        self.alpha = 1
        self.max_iter = 1e5  # 10000
        self.error_region = 1e-5  # 0.00001

        # M-estimators
        self.resid_sigmas_edge = []
        self.resid_sigmas_planes = []
        self.use_estimators = use_estimators
        self.initial_sigma_scale = 3
        self.sigma_coef = region_rate

        # Visualization
        self.visualize = False

    def fit(self, edge_corresp, plane_corresp, initial_T=None):
        T = mrob.geometry.SE3(np.zeros(6)) if not initial_T else initial_T  # w=T, wo: init_T=I
        #print(T.T())
        iter_cnt = 0

        initial_err = self.cost_function(edge_corresp, plane_corresp, T)  # init results
        errors = [initial_err + 1, initial_err]

        inlier_errors = [initial_err + 1, initial_err]
        outlier_errors = []
        inliers_edge_perc = []
        inliers_plane_perc = []
        inlier_edge_errors = []
        inlier_plane_errors = []

        if self.use_estimators:
            for corresp in edge_corresp:
                r, _ = self.r(corresp, T, corresp_type='edge')
                self.resid_sigmas_edge.append(self.initial_sigma_scale * self.e(r))

            for corresp in plane_corresp:
                r, _ = self.r(corresp, T, corresp_type='plane')
                self.resid_sigmas_planes.append(self.initial_sigma_scale * self.e(r))

        prev_rel_inlier_err = 100
        rel_inlier_err = 0

        prev_inliers_plane = []
        prev_inliers_edge = []

        while iter_cnt < self.max_iter and abs(prev_rel_inlier_err - rel_inlier_err) > self.error_region:
            prev_rel_inlier_err = rel_inlier_err
            jac, hess, edge_inliers, plane_inliers = self.derivatives(edge_corresp, plane_corresp, T)#寻找inlier点
            inliers_edge_perc.append(len(edge_inliers) / len(edge_corresp)) # 求比例？
            inliers_plane_perc.append(len(plane_inliers) / len(plane_corresp)) # 求比例？
            #hessian matrix的条件数（直接算2-norm）

            #更新T矩阵
            if np.linalg.cond(hess) < 1 / sys.float_info.epsilon: #矩阵条件数
                T.update_lhs(-self.alpha * np.linalg.inv(hess) @ jac.T) #
            else:
                break

            #更新error
            prev_inliers_edge = edge_inliers  # 点索引序号
            prev_inliers_plane = plane_inliers # 点索引序号
            errors.append(self.cost_function(edge_corresp, plane_corresp, T)) # 上面更新了T，新的error会减小

            edge_outliers = [x for x in range(len(edge_corresp)) if x not in edge_inliers]
            plane_outliers = [x for x in range(len(plane_corresp)) if x not in plane_inliers]
            outlier_errors.append(self.cost_function_by_ind(edge_corresp, plane_corresp, T,
                                                            edge_outliers, plane_outliers))

            inlier_errors.append(self.cost_function_by_ind(edge_corresp, plane_corresp, T,
                                                           edge_inliers, plane_inliers))

            inlier_edge_errors.append(self.cost_function_by_ind(edge_corresp, plane_corresp, T,
                                                                edge_inliers, []))

            inlier_plane_errors.append(self.cost_function_by_ind(edge_corresp, plane_corresp, T,
                                                                 [], plane_inliers))

            rel_inlier_err = inlier_errors[-1] / (len(edge_inliers) + len(plane_inliers))

            iter_cnt += 1

            if self.use_estimators:
                self.resid_sigmas_edge = [self.sigma_coef * x if x > self.error_region else x for x
                                          in self.resid_sigmas_edge]
                self.resid_sigmas_planes = [self.sigma_coef * x if x > self.error_region else x for x
                                            in self.resid_sigmas_planes]
        print(iter_cnt)
        return T, inlier_errors, prev_inliers_edge, prev_inliers_plane

    #判断是否满足inlier条件
    #原论文估计 特征点 到 参考(线/面)距离
    def m_estimator_condition(self, r, resid_sigma):
        if self.e(r) < resid_sigma:
            # aaa=self.e(r)
            return 1
        else:
            return 0
    '''
    r是距离
    基于zhang17: De=|(p-a)@(p-b)| / |a-b| # 面积/底  分子是有abp构成三角形面积的2倍; 分子
                Dp=(p-a) [(a-b)(a-c)] / |(a-b)(a-c)| # 其中第一项可以是 p-a/b/c,因为 [(a-b)(a-c)] / |(a-b)(a-c)|是单位法向
    下面的实现:  De=|(a-b)@(a-p)| / |a-b|   等效       
    '''
    def r(self, corresp, T, corresp_type=None):
        if corresp_type == 'edge':  # 0=feature points ; 1 2 = correspondence line
            a = corresp[1]  # point 1 of line
            b = corresp[2]  # point 2 of line
            p = T.transform_array(corresp[0].reshape(1, 3)).reshape(3)  # rotated edge feature point
            # r=np.dot(self._skew(a - b),(a - p).reshape(3, 1)) / np.linalg.norm(a - b) # 3rd
            r = self._skew(a - b) @ (a - p).reshape(3, 1) / np.linalg.norm(a - b) # 2nd
            # r=self._skew(a - b).dot((a - p).reshape(3, 1)) / np.linalg.norm(a - b) # 1st
            small_jac = self._skew(a - b) @ np.hstack((self._skew(p), -np.eye(3))) / np.linalg.norm(a - b)
            return r, small_jac
        elif corresp_type == 'plane':  # 0=feature points ; 1 2 3 = correspondence planner patch
            a = corresp[1]  # point 1 of planner
            b = corresp[2]  # point 2 of planner
            c = corresp[3]  # point 3 of planner
            p = T.transform_array(corresp[0].reshape(1, 3)).reshape(3)  # rotated planner feature point
            cross_prod = self._skew(a - b) @ (a - c).reshape(3, 1)
            n = cross_prod / np.linalg.norm(cross_prod)
            r = n.reshape(1, 3) @ (a - p).reshape(3, 1)
            small_jac = n.reshape(1, 3) @ np.hstack((self._skew(p), -np.eye(3)))
            return r, small_jac
        else:
            raise 'Type error'# NotSupportedType("NotSupportedType")
    #1*n dot n*1 = 3*3
    def e(self, r):  # dot pruduct r.r
        # r=r.reshape(1, -1)[0]
        # d=np.einsum('i,i', r,r)# slow
        return (r.reshape(1, -1) @ r.reshape(-1, 1))[0, 0]  # fast

    def cost_function(self, edge_corresp, plane_corresp, T):
        err = 0

        for corresp in edge_corresp:  # for each 3-3 array
            r, _ = self.r(corresp, T, corresp_type='edge')#距离=点到直线距离
            err += self.e(r)#坐标点乘=距离误差

        for corresp in plane_corresp:  # for each 4-3 array
            r, _ = self.r(corresp, T, corresp_type='plane')#点到平面距离
            err += self.e(r)#坐标点乘=距离误差

        return err

    def cost_function_by_ind(self, edge_corresp, plane_corresp, T, edge_ind, plane_ind):
        err = 0
        for i in edge_ind:
            r, _ = self.r(edge_corresp[i], T, corresp_type='edge')
            err += self.e(r)

        for i in plane_ind:
            r, _ = self.r(plane_corresp[i], T, corresp_type='plane')
            err += self.e(r)

        return err

    #生成inlier_edge 和 inlier_plane
    def derivatives(self, edge_corresp, plane_corresp, T):
        jac = np.zeros((1, 6)) # 1*6
        hess = np.zeros((6, 6)) #6*6
        #寻找 inlier 和 outlier点，原文zhang17是基于 点 到 correspondences的距离
        #edge line as the correspondence for an edge point
        edge_inliers = []
        for ind, corresp in enumerate(edge_corresp):
            r, jac_i = self.r(corresp, T, corresp_type='edge')
            if not self.use_estimators or ( # 不用est 或者 est,但不满足条件
                    self.use_estimators and self.m_estimator_condition(r, self.resid_sigmas_edge[ind])):
                jac += r.T @ jac_i #1*3 @ 3*6 = 1*6
                hess += jac_i.T @ jac_i #6*1 @ 1*6
                edge_inliers.append(ind)
        #planar patch as the correspondence for a planar point
        plane_inliers = []
        for ind, corresp in enumerate(plane_corresp):
            r, jac_i = self.r(corresp, T, corresp_type='plane') #这里使用 ja
            if not self.use_estimators or (
                    self.use_estimators and self.m_estimator_condition(r, self.resid_sigmas_planes[ind])):
                jac += r.T @ jac_i #jacobian matrix
                hess += jac_i.T @ jac_i #hessian matrix
                plane_inliers.append(ind)

        return jac, hess, edge_inliers, plane_inliers

    #skew symmetric matrix
    #@x 1*3 array to ssm
    def _skew(self, x):
        #print(x)
        return np.array([[0, -x[2], x[1]],
                         [x[2], 0, -x[0]],
                         [-x[1], x[0], 0]])

    # def NotSupportedType(self):
    #    print("Not Support Type.")
