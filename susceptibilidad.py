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
file = input('Nombre del archivo: ')
data = pd.read_csv(file)
f = data.iloc[:,0]
f = pd.to_numeric(f)
x = data.iloc[:,1]
x = pd.to_numeric(x)
y = data.iloc[:,2]
y = pd.to_numeric(y)

cero = pd.read_csv('cero_csv.csv')
f_cero = cero.iloc[:,0]
f_cero = pd.to_numeric(f_cero)
x_cero = cero.iloc[:,1]
x_cero = pd.to_numeric(x_cero)
y_cero = cero.iloc[:,2]
y_cero = pd.to_numeric(y_cero)

for i in range(397):
    x[i] = x[i] - x_cero[i]
    y[i] = y[i] - y_cero[i]

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(f, x, label='Re(V)')
plt.plot(f, y, label='Im(V)')
plt.grid(True)
plt.legend()
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Voltaje (V)')
plt.ylim(-0.002, 0.008)
plt.xlim(0, 22500)
plt.show()

#%%
z = np.zeros(397)

for i in range(397):
    z[i] = -y[i]/x[i]
    
#figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
#plt.plot(f, z)
#plt.grid(True)
#plt.show()

#%%
X = np.zeros(397)

for i in range(397):
    X[i] = -0.01 + 3.06*z[i] - 0.105*z[i]**2 + 0.167*z[i]**3

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(f, X)
plt.grid(True)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('X')
plt.show()

#%%
X_lin = []
f_lin = []
for i in range (397):
    if X[i] > 1 and X[i] < 10:
        X_lin.append(X[i])
        f_lin.append(f[i])
X_lin = np.array(X_lin)
f_lin = np.array(f_lin)

r = float(input('Radio de la muestra (en metros): '))

def func_X(f, rho):
    return 0.00000395*(r**2)*f/rho

p_opt, p_cov = curve_fit(func_X, f_lin, X_lin, [0.0000000282])
print('Pendiente del ajuste: ', p_opt)
print('Dev std: ', np.sqrt(p_cov[0]))

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(f_lin, X_lin, label='X')
plt.plot(f_lin, func_X(f_lin, p_opt), label='Ajuste lineal')
plt.grid(True)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('X')
plt.legend()
plt.show()