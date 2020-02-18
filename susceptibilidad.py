# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:03:24 2020

@author: Bruno
"""
#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from scipy.optimize import curve_fit
from scipy import special

#%%
lat1 = pd.read_csv('lat1_csv.csv')
f_lat1 = lat1.iloc[:,0]
f_lat1 = pd.to_numeric(f_lat1)
x_lat1 = lat1.iloc[:,1]
x_lat1 = pd.to_numeric(x_lat1)
y_lat1 = lat1.iloc[:,2]
y_lat1 = pd.to_numeric(y_lat1)

cero = pd.read_csv('cero_csv.csv')
f_cero = cero.iloc[:,0]
f_cero = pd.to_numeric(f_cero)
x_cero = cero.iloc[:,1]
x_cero = pd.to_numeric(x_cero)
y_cero = cero.iloc[:,2]
y_cero = pd.to_numeric(y_cero)

for i in range(397):
    x_lat1[i] = x_lat1[i] - x_cero[i]
    y_lat1[i] = y_lat1[i] - y_cero[i]

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(f_lat1, x_lat1)
plt.plot(f_lat1, y_lat1)
plt.grid(True)
plt.ylim(-0.002, 0.008)
plt.xlim(0, 22500)
plt.show()

#%%
z_lat1 = np.zeros(397)

for i in range(397):
    z_lat1[i] = -y_lat1[i]/x_lat1[i]
    
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(f_lat1, z_lat1)
plt.grid(True)
plt.show()

#%%
X_lat1 = np.zeros(397)

for i in range(397):
    X_lat1[i] = -0.01 + 3.06*z_lat1[i] - 0.105*z_lat1[i]**2 + 0.167*z_lat1[i]**3

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(f_lat1, X_lat1)
plt.grid(True)
plt.show()

#%%
X_lat1_lin = X_lat1[11:116]
f_lat1_lin = f_lat1[11:116]

def func_x(f, rho):
    return 0.00000395*0.000042706225*f/rho

p_opt, p_cov = curve_fit(func_x, f_lat1_lin, X_lat1_lin, [0.0000000282])
print(p_opt)
print(np.sqrt(p_cov[0]))

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(f_lat1_lin, X_lat1_lin)
plt.plot(f_lat1_lin, func_x(f_lat1_lin, p_opt))
plt.grid(True)
plt.show()