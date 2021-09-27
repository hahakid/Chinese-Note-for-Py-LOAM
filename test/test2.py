import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt


def real_func(x):
    return np.sin(2*np.pi*x)

def fit_func(p,x):
    f=np.poly1d(p)
    return f(x)

def residuals_func(p,x,y):
    ret=fit_func(p,x)-y
    return ret

x = np.linspace(0, 1, 10)
x_points = np.linspace(0, 1, 1000)
y_ = real_func(x)
y = [np.random.normal(0, 0.1)+y1 for y1 in y_]

regularization = 0.0001 #正则项
def residuals_func_regularization(p, x, y):
    ret = fit_func(p, x) - y
    ret = np.append(ret, np.sqrt(regularization * np.abs(p)))  # L1范数作为正则化项
    #ret = np.append(ret, np.sqrt(0.5*regularization*np.square(p))) # L2范数作为正则化项
    return ret

def fitting(M=0):
    p_init=np.random.rand(M+1)
    p_lsq=leastsq(residuals_func,p_init,args=(x,y))
    p_lsq_regularization = leastsq(residuals_func_regularization, p_init, args=(x, y))
    print(p_init)
    #print('Fitting Parameters:', p_lsq[0])
    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve',linewidth=3)
    plt.plot(x_points, fit_func(p_lsq_regularization[0], x_points), label='regularization')
    plt.plot(x, y, 'bo', label='noise')
    plt.legend()
    plt.show()
    return p_lsq

for i in range(0,10):
    #用不同元函数进行拟合
    p_lsq = fitting(i)