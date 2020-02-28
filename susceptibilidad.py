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

plt.rcParams.update({'font.size': 15})
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

err_f = np.zeros(len(data))

for i in range(len(data)):
    x[i] = x[i] - x_cero[i]
    y[i] = y[i] - y_cero[i]
    err_f[i] = f[i]*0.01

err_x = np.zeros(len(data))
err_y = np.zeros(len(data))

for i in range(len(data)):
    err_x[i] = x[i]*0.02
    err_y[i] = y[i]*0.02

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.errorbar(f, x, err_x, None, '.', errorevery=5, label='Re(V)')
plt.errorbar(f, y, err_y, None, '.', errorevery=5, label='Im(V)')
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

err_z = np.zeros(len(data))

for i in range(len(data)):
    err_z[i]= np.sqrt((y[i]/x[i]**2)**2 * err_x[i]**2 + (-1/x[i])**2 * err_y[i]**2)

#%%defino X
X = np.zeros(len(data))

for i in range(len(data)):
    X[i] = -0.01 + 3.06*z[i] - 0.105*z[i]**2 + 0.167*z[i]**3
    
err_X = np.zeros(len(data))

for i in range(len(data)):
    err_X[i] = np.sqrt((3.06 - 2*0.105*z[i] + 3*0.167*z[i]**2)**2 * err_z[i]**2)

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.errorbar(f, X, err_X, None, '.', errorevery=5)
plt.grid(True)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('X')
plt.show()

#%%busco el rango en que X es mayor a 1 y menor a 10 y le ajusto una lineal
X_lin = []
f_lin = []
err_X_lin = []
index = []
err_f_lin = []

for i in range(len(data)):
    if X[i] > 1 and X[i] < 10:
        index.append(i)
        X_lin.append(X[i])
        f_lin.append(f[i])
        err_X_lin.append(err_X[i])
        err_f_lin.append(err_f[i])
        
X_lin = np.array(X_lin)
f_lin = np.array(f_lin)
err_X_lin = np.array(err_X_lin)
err_f_lin = np.array(err_f_lin)

r = float(input('Radio de la muestra (en metros): '))

def func_X(f, rho):
    return 0.00000395*(r**2)*f/rho

p_opt, p_cov = curve_fit(func_X, f_lin, X_lin, [0.0000000282], sigma=err_X_lin, absolute_sigma=True)
print('Pendiente del ajuste: ', p_opt)
print('Dev std: ', np.sqrt(p_cov[0]))

rho = p_opt

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.errorbar(f_lin, X_lin, err_X_lin, None, '.', errorevery=1, label='X')
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
err_x_lin = err_x[a_lin:b_lin+1]
err_y_lin = err_y[a_lin:b_lin+1]

for i in range(len(f_lin)):
    chi_p[i] = -(y_lin[i])/(2*np.pi*f_lin[i])
    chi_pp[i] = (x_lin[i])/(2*np.pi*f_lin[i])

chi_p = abs(chi_p)
chi_pp = abs(chi_pp)

err_chi_p = np.zeros(len(f_lin))
err_chi_pp = np.zeros(len(f_lin))

for i in range(len(f_lin)):
    err_chi_p[i] = np.sqrt((1/(2*np.pi*f_lin[i]))**2 * err_y_lin[i]**2 + (y_lin[i]/(2*np.pi*f_lin[i]**2))**2 * err_f_lin[i]**2)
    err_chi_pp[i] = np.sqrt((1/(2*np.pi*f_lin[i]))**2 * err_x_lin[i]**2 + (x_lin[i]/(2*np.pi*f_lin[i]**2))**2 * err_f_lin[i]**2)

def func_chi_p(X, k):
    return k*(-0.088 + 0.1503*X + 0.01566*X**2 - 0.00737*X**3 + 0.0007755*X**4 - 0.00002678*X**5)

def func_chi_pp(X, k):
    return k*(-0.048 + 0.378*X - 0.12207*X**2 + 0.017973*X**3 - 0.0012777*X**4 + 0.00003542*X**5)

p_opt_pol_p, p_cov_pol_p = curve_fit(func_chi_p, X_lin, chi_p, sigma=err_chi_p, absolute_sigma=True)
print('k_p ', p_opt_pol_p, np.sqrt(p_cov_pol_p[0]))

p_opt_pol_pp, p_cov_pol_pp = curve_fit(func_chi_pp, X_lin, chi_pp, sigma=err_chi_pp, absolute_sigma=True)
print('k_pp ', p_opt_pol_pp, np.sqrt(p_cov_pol_pp[0]))

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.errorbar(X_delta, chi_p, err_chi_p, None, '.', errorevery=1, label='chi_p')
plt.errorbar(X_delta, chi_pp, err_chi_p, None, '.', errorevery=1, label='chi_pp')
plt.plot(X_delta, func_chi_p(X_lin, p_opt_pol_p), label='Ajuste chi_p')
plt.plot(X_delta, func_chi_pp(X_lin, p_opt_pol_pp), label='Ajuste chi_pp')
plt.grid(True)
plt.xlabel('r^2/delta^2')
plt.ylabel('Susceptibilidad')
plt.legend()
plt.show()

#%%Bessel
chi = np.zeros(len(f_lin), dtype=np.complex128)

for i in range(len(f_lin)):
    chi[i] = complex(chi_p[i]/p_opt_pol_p, chi_pp[i]/p_opt_pol_pp)

chi_bessel = np.zeros(len(f_lin), dtype=np.complex128)

for i in range(len(f_lin)):
    chi_bessel[i] = (2*(special.jv(1, (complex(1,1)/delta_lin[i])*r))/((complex(1,1)/delta_lin[i])*r * special.jv(0, (complex(1,1)/delta_lin[i])*r))) -1

mod_chi_bessel = abs(chi_bessel)
real_chi_bessel = chi_bessel.real
imag_chi_bessel = chi_bessel.imag

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.errorbar(X_delta, chi_p/p_opt_pol_p, err_chi_p/p_opt_pol_p, None, '.', errorevery=1, label='chi_p')
plt.errorbar(X_delta, chi_pp/p_opt_pol_pp, err_chi_pp/p_opt_pol_p, None, '.', errorevery=1, label='chi_pp')
plt.plot(X_delta, abs(real_chi_bessel), label='Re(chi_bessel)')
plt.plot(X_delta, abs(imag_chi_bessel), label='Im(chi_bessel)')
plt.grid(True)
plt.xlabel('r^2/delta^2')
plt.ylabel('Susceptibilidad')
plt.legend()
plt.show()
              
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(X_delta, abs(chi), '.', label='chi')
plt.plot(X_delta, mod_chi_bessel, '.', label='chi_bessel')
#plt.plot(X_delta, abs(real_chi_bessel), '.', label='Re(chi_bessel)')
#plt.plot(X_delta, abs(imag_chi_bessel), '.', label='Im(chi_bessel)')
#plt.plot(X_delta, func_chi_p(X_lin, 1),)
#plt.plot(X_delta, func_chi_pp(X_lin, 1),)
plt.grid(True)
plt.xlabel('r^2/delta^2')
plt.ylabel('Susceptibilidad')
plt.legend()
plt.show()

#%%
delta = np.zeros(len(f))
X_delta_total = np.zeros(len(f))
chi_p_total = np.zeros(len(f))
chi_pp_total = np.zeros(len(f))
chi_total = np.zeros(len(f), dtype=np.complex128)
chi_bessel_total = np.zeros(len(f), dtype=np.complex128)

for i in range(len(f)):
    delta[i] = np.sqrt(2*rho/(0.0000004*np.pi*2*np.pi*f[i]))
    X_delta_total[i] = r**2/(delta[i]**2)

for i in range(len(f)):
    chi_p_total[i] = -(y[i])/(2*np.pi*f[i])
    chi_pp_total[i] = (x[i])/(2*np.pi*f[i])

for i in range(len(f)):
    chi_total[i] = complex(chi_p_total[i]/p_opt_pol_p, chi_pp_total[i]/p_opt_pol_pp)
    
for i in range(len(f)):
    chi_bessel_total[i] = (2*(special.jv(1, (complex(1,1)/delta[i])*r))/((complex(1,1)/delta[i])*r * special.jv(0, (complex(1,1)/delta[i])*r))) -1

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(X[10:50], abs(chi_p_total/p_opt_pol_p)[10:50], '.', label='chi_p_total')
plt.plot(X[10:50], abs(chi_bessel_total.real)[10:50], label='Re(chi_bessel_total)')
plt.plot(X[10:50], abs(func_chi_p(X,1))[10:50], label='Polinomio_pp')
plt.plot(X[10:50], abs(chi_pp_total/p_opt_pol_pp)[10:50], '.', label='chi_pp_total')
plt.plot(X[10:50], abs(chi_bessel_total.imag)[10:50], label='Im(chi_bessel_total)')
plt.plot(X[10:50], abs(func_chi_pp(X,1))[10:50], label='Polinomio_p')
plt.grid(True)
plt.xlabel('r^2/delta^2')
plt.ylabel('Susceptibilidad')
plt.legend()
plt.show()