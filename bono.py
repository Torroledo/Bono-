#Importar librerias
import numpy as np
import sys,codecs,math
import os
from numpy import mean,cov,double,cumsum,dot,array,rank
from pylab import plot,subplot,axis,stem,show,figure
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pylab
from scipy.fftpack import fft, fftfreq
## PARTE 1:
M=1
r_ext=5
r0=1 
G=1
N=100
i=0
r=np.empty(100)
r_theta=np.empty(100)
x=np.empty(100)
y=np.empty(100)
v_theta=np.empty(100)
v=np.empty(100)

def velocidad(G,M,r):
    return math.sqrt(G*M/r)

#radios iniciales
#velocidades iniciales
for i in range(0,N-1):
    r[i]=r_ext+r0*i
    v[i]= velocidad(G, M, r[i])
    r_theta[i]=np.random.random()*2*3.1415
    v_theta[i]=np.random.random()*2*3.1415
    print r[i],v[i],r_theta[i],v_theta[i]

#conversion a XY
for i in range(N):
    x[i]=r[i]*math.cos(r_theta[i])
    y[i]=r[i]*math.sin(r_theta[i])
#    print x[i], y[i]
plt.scatter(x,y)
plt.show()

## PARTE 2:
def dr1dt(t,y_2):
    return y_2

def dr2dt(t, y_1, y_2):
	G=1.0
	M=1.0
    return -G*M/y_1**2

def EDO_solution(r,v,r_theta, v_theta):
    h=0.05
    
    xo=0.0;
    xf=1.0;
    n_points = int((xf-x0)/h)
    t = empty(n_points)
    r_1 = empty(n_points)
    r_2 = empty(n_points)
    t_old=0
    r1__old=r
    r2_old=v
    for i in range(1,n_points):
        t_old=t[i-1]
        r1_old=r1[i-1]
        r2_old=r2[i-1]
        
        k1_r1=dr1dt(t_old,r2_old)
        k1_r2=dr2dt(t_old,r1_old,r2_old)
        t1=t_old+(h/2.0)
        r1_step1=r1_old + (h/2.0) * k1_r1
        r2_step1=r2_old + (h/2.0) * k1_r2
        
        k2_r1=dr1dt(t1,r2_step1)
        k2_r2=dr2dt(t1,r1_step1,r2_step1)
        t2=t_old+(h/2.0)
        r1_step2=r1_old + (h/2.0) * k2_r1
        r2_step2=r2_old + (h/2.0) * k2_r2
        
        k3_r1=dr1dt(t1,r2_step2)
        k3_r2=dr2dt(t2,r1_step2,r2_step2)
        t3=t_old+(h/2.0)
        r1_step3=r1_old + (h/2.0) * k3_r1
        r2_step3=r2_old + (h/2.0) * k3_r2
        
        k4_r1=dr1dt(t3,r2_step3)
        k4_r2=dr2dt(t3,r1_step3,r2_step3)
        t3=t_old+(h/2.0)
        r1_step3=r1_old + (h/2.0) * k3_r1
        r2_step3=r2_old + (h/2.0) * k3_r2
        
        slope_r1=(1.0/6.0)*(k1_r1+2.0*k2_r1+2.0*k3_r1+k4_r1);
        slope_r2=(1.0/6.0)*(k1_r2+2.0*k2_r2+2.0*k3_r2+k4_r2);
        
        t[i+1]=t_old+h;
        r1[i+1]=r1_old+h*slope_r1;
        r2[]i+1=r2_old+h*slope_r2;
    return r1

for i in range(0,100):
r_estrellas[i]=EDO_soluition(r0[i],v0[i])
temp=r_estrellas[i]











