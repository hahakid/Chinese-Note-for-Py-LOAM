import numpy as np
import open3d as o3d
path='/media/kid/data/ov/sequences/00/velodyne/000000.bin'
pcd = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(pcd[:, :3])

print(0.00137**2+0.05402**2+0.011**2)

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

a=np.random.randint(0,high=2,size=pcd.shape[0])
#print(a)
#cl,ind=pc.remove_statistical_outlier(20,2)
count=0
ind=[]
for i in a:
    count+=i
    if i != 0:
        ind.append(count)
sss=pc.select_by_index(ind)
sss.paint_uniform_color([1.0,0,0])
pc_show([sss])
#b=np.linalg.norm(a, 2)
#print(b)

#print(np.convolve([1, 2, 3], [-1, 1, 1]))
#print(np.convolve([1, 2, 3], [0, 1, 0.5]))





