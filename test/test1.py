# coding: utf-8
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

#import random
#Xi=np.array([6.19,2.51,7.29,7.01,5.7,2.66,3.98,2.5,9.1,4.2])
#Yi=np.array([5.25,2.83,6.41,6.71,5.1,4.23,5.05,1.98,10.5,6.3])
Xi=np.array([1,2,3,4,5,6])
#Yi=np.array([9,18,31,48,69,94])
Yi=np.array([9.1,18.3,32,47,69.5,94.8])
#Xi=np.random.randn(100)
#Yi=np.random.randn(100)
print(Xi,Yi)

def func(p,x):
    k,b=p
    return k*(x**2)+b

def func1(p,x):
    a,b,c=p
    return a*(x**2)+b*x+c

def error(p,x,y):
    return func(p,x)-y

def error1(p,x,y):
    return func1(p,x)-y

p0=[1,20]
Para=leastsq(error,p0,args=(Xi,Yi))

p1=[1,1,1]
Para1=leastsq(error1,p1,args=(Xi,Yi))

k,b=Para[0]
print("k=",k,"b=",b)
print("cost："+str(Para[1]))
print("求解的拟合直线为:")
print("y="+str(round(k,2))+"x^2+"+str(round(b,2)))

plt.figure(figsize=(8,8)) ##指定图像比例： 8：6
plt.scatter(Xi,Yi,color="green",label="sample",linewidth=2)

#画拟合直线
x=np.linspace(0,12,100) ##在0-15直接画100个连续点
y=k*(x**2)+b ##函数式
plt.plot(x,y,color="red",label="line",linewidth=5)


a,b,c=Para1[0]
x1=np.linspace(0,12,100) ##在0-15直接画100个连续点
y1=a*(x**2)+b*x+c ##函数式
plt.plot(x,y,color="green",label="line",linewidth=1)

plt.legend(loc='lower right') #绘制图例
plt.show()





