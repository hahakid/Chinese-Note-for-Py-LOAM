import mrob
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

#plt.rcParams['pdf.fonttype'] = 42
#plt.rcParams['ps.fonttype'] = 42

aaa=np.asarray([[0,2,3],[4,0,6],[7,8,0]])

x0=aaa[:,0]
x1=aaa[:,1]
x2=aaa[:,2]


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])
print(x0,x1,x2)
d1=np.linalg.norm(skew(x2 - x1) @ (x1 - x0).reshape(3, 1))
d2=np.linalg.norm(skew(x0 - x1) @ (x0 - x2).reshape(3, 1))
d3=np.linalg.norm(skew(x1 - x2) @ (x1 - x0).reshape(3, 1))
d4=np.linalg.norm(skew(x0 - x2) @ (x1 - x0).reshape(3, 1))

print(d1)
print(d2)
print(d3)
print(d4)
#bbb=-aaa.T
#print(bbb)


def plotConfig():
    "configfures the 3d plot structure for representing tranformations"
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    return ax


def plotT(T, ax):
    "Plots a 3 axis frame in the origin given the mrob SE3 transformation, right-hand convention"
    # transform 3 axis to the coordinate system
    x = np.zeros((4, 3))
    x[0, :] = T.transform(np.array([0, 0, 0], dtype='float64'))
    x[1, :] = T.transform(np.array([1, 0, 0], dtype='float64'))
    ax.plot(x[[0, 1], 0], x[[0, 1], 1], x[[0, 1], 2], 'r')  # X axis
    x[2, :] = T.transform(np.array([0, 1, 0], dtype='float64'))
    ax.plot(x[[0, 2], 0], x[[0, 2], 1], x[[0, 2], 2], 'g')  # Y axis
    x[3, :] = T.transform(np.array([0, 0, 1], dtype='float64'))
    ax.plot(x[[0, 3], 0], x[[0, 3], 1], x[[0, 3], 2], 'b')  # Z axis
    plt.xlabel('x')
    plt.ylabel('y')

rotation_angles = np.random.rand(3)
R = mrob.geometry.SO3(rotation_angles)

translation_vector = np.array([5, 5, 5])
T = np.eye(4)
T[:3, :3] = R.R()
T[:, 3][:-1] = translation_vector #5,5,5
T = mrob.geometry.SE3(T)

#ax = plotConfig()
#plotT(T, ax)
#plt.title('Rotation from bare-hands')
#plt.show()

xi = mrob.geometry.SE3(np.concatenate([rotation_angles, translation_vector]))
T2 = mrob.geometry.SE3(xi)

print(T.T())
print(" ")
print(T2.T())

ax = plotConfig()
plotT(T, ax)
plotT(T2, ax)
plt.title('Rotation from direct mapping')
plt.show()












