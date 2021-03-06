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
            jac, hess, edge_inliers, plane_inliers = self.derivatives(edge_corresp, plane_corresp, T)#??????inlier???
            inliers_edge_perc.append(len(edge_inliers) / len(edge_corresp)) # ????????????
            inliers_plane_perc.append(len(plane_inliers) / len(plane_corresp)) # ????????????
            #hessian matrix????????????????????????2-norm???

            #??????T??????
            if np.linalg.cond(hess) < 1 / sys.float_info.epsilon: #???????????????
                T.update_lhs(-self.alpha * np.linalg.inv(hess) @ jac.T) #
            else:
                break

            #??????error
            prev_inliers_edge = edge_inliers  # ???????????????
            prev_inliers_plane = plane_inliers # ???????????????
            errors.append(self.cost_function(edge_corresp, plane_corresp, T)) # ???????????????T?????????error?????????

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

    #??????????????????inlier??????
    #??????????????? ????????? ??? ??????(???/???)??????
    def m_estimator_condition(self, r, resid_sigma):
        if self.e(r) < resid_sigma:
            # aaa=self.e(r)
            return 1
        else:
            return 0
    '''
    r?????????
    ??????zhang17: De=|(p-a)@(p-b)| / |a-b| # ??????/???  ????????????abp????????????????????????2???; ??????
                Dp=(p-a) [(a-b)(a-c)] / |(a-b)(a-c)| # ???????????????????????? p-a/b/c,?????? [(a-b)(a-c)] / |(a-b)(a-c)|???????????????
    ???????????????:  De=|(a-b)@(a-p)| / |a-b|   ??????       
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
            r, _ = self.r(corresp, T, corresp_type='edge')#??????=??????????????????
            err += self.e(r)#????????????=????????????

        for corresp in plane_corresp:  # for each 4-3 array
            r, _ = self.r(corresp, T, corresp_type='plane')#??????????????????
            err += self.e(r)#????????????=????????????

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

    #??????inlier_edge ??? inlier_plane
    def derivatives(self, edge_corresp, plane_corresp, T):
        jac = np.zeros((1, 6)) # 1*6
        hess = np.zeros((6, 6)) #6*6
        #?????? inlier ??? outlier????????????zhang17????????? ??? ??? correspondences?????????
        #edge line as the correspondence for an edge point
        edge_inliers = []
        for ind, corresp in enumerate(edge_corresp):
            r, jac_i = self.r(corresp, T, corresp_type='edge')
            if not self.use_estimators or ( # ??????est ?????? est,??????????????????
                    self.use_estimators and self.m_estimator_condition(r, self.resid_sigmas_edge[ind])):
                jac += r.T @ jac_i #1*3 @ 3*6 = 1*6
                hess += jac_i.T @ jac_i #6*1 @ 1*6
                edge_inliers.append(ind)
        #planar patch as the correspondence for a planar point
        plane_inliers = []
        for ind, corresp in enumerate(plane_corresp):
            r, jac_i = self.r(corresp, T, corresp_type='plane') #???????????? ja
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
