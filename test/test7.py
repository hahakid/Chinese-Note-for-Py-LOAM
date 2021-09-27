import numpy as np
import open3d as o3d
from LOAM import utils

def pc_show(pc,norm_flag=False):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800,height=800)
    opt = vis.get_render_option()
    opt.point_size = 2
    opt.point_show_normal=norm_flag
    for p in pc:
        vis.add_geometry(p)
    vis.run()
    vis.destroy_window()

#s = np.asarray([[0,0,0],[0,1,0],[1,0,0],[-1,0,0],[0,-1,0]])  # 不满足条件
#s = np.asarray([[0,0,0],[0,1,0],[0,1,1],[0,0,1],[0,0.5,0.5]])
s = np.asarray([[1,0,0],[0,1,0],[0,0,1],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]])
b = np.ones((s.shape[0], ))
norm = np.linalg.lstsq(s, b, rcond=None)[0]
norm_reversed = 1 / np.linalg.norm(norm)
norm /= np.linalg.norm(norm)
flag=True
for i in range(0,s.shape[0]):
    #print(s[i,:])
    if np.abs(np.dot(norm, s[i,:]) + norm_reversed) > 0.2:
        flag=False
        break
if flag:
    print(norm,norm_reversed)
centroid=np.sum(s, axis=0) / s.shape[0]


pcd=np.vstack([centroid,s])
pcd=utils.get_pcd_from_numpy(pcd)
#pc_show([pcd])

pc = utils.get_pcd_from_numpy(s)
plane_model, inliers = pc.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=50)

print(norm, norm_reversed, plane_model)
