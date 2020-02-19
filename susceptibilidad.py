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

#%%importo los datos y creo mis tiras de datos
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
plt.plot(f, x, '.', label='Re(V)')
plt.plot(f, y, '.', label='Im(V)')
plt.grid(True)
plt.legend()
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Voltaje (V)')
plt.ylim(-0.002, 0.008)
plt.xlim(0, 22500)
plt.show()

#%%defino z
z = np.zeros(len(data))

for i in range(len(data)):
    z[i] = -y[i]/x[i]

#%%defino X
X = np.zeros(len(data))

for i in range(len(data)):
    X[i] = -0.01 + 3.06*z[i] - 0.105*z[i]**2 + 0.167*z[i]**3

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(f, X, '.')
plt.grid(True)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('X')
plt.show()

#%%busco el rango en que X es mayor a 1 y menor a 10 y le ajusto una lineal
X_lin = []
f_lin = []
index = []
for i in range(len(data)):
    if X[i] > 1 and X[i] < 10:
        index.append(i)
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

rho = p_opt

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(f_lin, X_lin, '.', label='X')
plt.plot(f_lin, func_X(f_lin, rho), label='Ajuste lineal')
plt.grid(True)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('X')
plt.legend()
plt.show()

#%%defino chi_p y chi_pp a partir de las mediciones (a menos de una constante) y ajusto los polinomios
chi_p = np.zeros(len(f_lin))
chi_pp = np.zeros(len(f_lin))
delta_lin = np.zeros(len(f_lin))
X_delta = np.zeros(len(f_lin))

for i in range(len(f_lin)):
    delta_lin[i] = np.sqrt(2*rho/(0.0000004*np.pi*2*np.pi*f_lin[i]))
    X_delta[i] = r**2/(delta_lin[i]**2)

a_lin = index[0]
b_lin = index[len(index)-1]
x_lin = x[a_lin:b_lin+1].to_numpy()
y_lin = y[a_lin:b_lin+1].to_numpy()

for i in range(len(f_lin)):
    chi_p[i] = -(y_lin[i])/(2*np.pi*f_lin[i])
    chi_pp[i] = (x_lin[i])/(2*np.pi*f_lin[i])

chi_p = np.absolute(chi_p)
chi_pp = np.absolute(chi_pp)


def func_chi_p(X):
    return -0.088 + 0.1503*X + 0.01566*X**2 - 0.00737*X**3 + 0.0007755*X**4 - 0.00002678*X**5

def func_chi_pp(X):
    return -0.048 + 0.378*X - 0.12207*X**2 + 0.017973*X**3 - 0.0012777*X**4 + 0.00003542*X**5


figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(X_delta, chi_p, '.', label='chi_p')
plt.plot(X_delta, chi_pp, '.', label='chi_pp')
plt.plot(X_delta, func_chi_p(X_lin), label='Ajuste chi_p')
plt.plot(X_delta, func_chi_pp(X_lin), label='Ajuste chi_pp')
plt.grid(True)
plt.xlabel('r^2/delta^2')
plt.ylabel('Susceptibilidad')
plt.legend()
plt.show()

#%%Bessel
chi = np.zeros(len(f_lin), dtype=np.complex128)

for i in range(len(f_lin)):
    chi[i] = complex(chi_p[i], chi_pp[i])

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(X_delta, chi, '.', label='chi')
plt.grid(True)
plt.xlabel('r^2/delta^2')
plt.ylabel('Susceptibilidad')
plt.legend()
plt.show()

chi_bessel = np.zeros(len(f_lin), dtype=np.complex128)

for i in range(len(f_lin)):
    chi_bessel[i] = 2*(special.jv(1, (complex(1,1)/delta_lin[i])*r))/((complex(1,1)/delta_lin[i])*r * special.jv(0, (complex(1,1)/delta_lin[i])*r)) -1

chi_bessel = np.absolute(chi_bessel)
              
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(X_delta, chi, '.', label='chi')
plt.plot(X_delta, chi_bessel, '.', label='chi_bessle')
plt.grid(True)
plt.xlabel('r^2/delta^2')
plt.ylabel('Susceptibilidad')
plt.legend()
plt.show()