import mrob
import numpy as np
import transforms3d
np.set_printoptions(precision=4, suppress=True)
# example equal to ./PC_alignment/examples/example_align.cpp
# generate random data
N = 1
#X = np.random.rand(N,3)
X=np.asarray([1,1,1])
#sixele = np.asarray([np.pi,np.pi,np.pi,1,1,1])# r,t #np.random.rand(6)
#sixele = np.asarray([-0.2,-np.pi/2,-0.2,0,0,0])# r,t #np.random.rand(6)
sixele = np.asarray([np.pi,np.pi,np.pi,0,0,0])
rx=np.asarray([np.pi/8,0,0,0,0,0])
ry=np.asarray([0,np.pi/6,0,0,0,0])
rz=np.asarray([0,0,np.pi/6,0,0,0])
rxyz=np.asarray([np.pi/6,np.pi/6,np.pi/6,0,0,0])
Tx = mrob.geometry.SE3(rx)
#Tx.print()
Ty = mrob.geometry.SE3(ry)
#Ty.print()
Tz = mrob.geometry.SE3(rz)
#Tz.print()
xyz = mrob.geometry.SO3(rxyz[0:3])
xyz_se = mrob.geometry.SE3(rxyz)

#Txyz.print()
#mrob.geometry.SE3(np.zeros(6)).print()
#Y = T.transform_array(X.reshape(1,3))

def rigid_translate(pc_input, extrinsic):
    # projection
    scan = np.row_stack([pc_input[:, :3].T, np.ones_like(pc_input[:, 0])])
    scan = np.matmul(extrinsic, scan)
    points = np.row_stack([scan[:3, :], pc_input[:, 3:].T]).T
    return points

def ex(r,t):
    r=np.asarray(r)
    t=np.asarray(t)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = transforms3d.euler.euler2mat(r[0],r[1],r[2]) # rotation
    extrinsic[:3, 3] = t.reshape(-1) # translation
    return extrinsic

def rigid_translate(pc_input, extrinsic):
    # projection
    scan = np.row_stack([pc_input[:, :3].T, np.ones_like(pc_input[:, 0])])
    scan = np.matmul(extrinsic, scan)
    points = np.row_stack([scan[:3, :], pc_input[:, 3:].T]).T
    return points

#print(sixele[0:3],sixele[3:6])
TT=ex(sixele[0:3],sixele[3:6])
YY=rigid_translate(X.reshape(1,3),TT)#np.matmul(TT, X.reshape(1,3))

#print(np.sin(180))
#print('X = \n', X,'\n T = \n', T.T(),'\n Y =\n', Y,'\n TT =\n', TT,'\n YY =\n', YY)
#print('\n T = \n', T.T(),'\n TT =\n', TT)
#print('\n Y =\n', Y,'\n YY =\n', YY)

TTx=ex(rx[0:3],rx[3:6])
print(TTx)
Tx.print()
TTy=ex(ry[0:3],ry[3:6])
print(TTy)
Ty.print()
TTz=ex(rz[0:3],rz[3:6])
print(TTz)
Tz.print()
Txyz=ex(rxyz[0:3],rxyz[3:6])
TTxyz=TTx@TTy@TTz

print(Txyz)
print(TTxyz)
xyz.print()