# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 08:00:21 2020

@author: Bruno
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from scipy.optimize import curve_fit
from scipy import special

data_res = pd.read_csv('Resonancia.csv')

f_datos_res = data_res.iloc[:,0]
f_datos_res = pd.to_numeric(f_datos_res)
v_in_datos_res = data_res.iloc[:,1]
v_in_datos_res = pd.to_numeric(v_in_datos_res)
v_out_datos_res = data_res.iloc[:,2]
v_out_datos_res = pd.to_numeric(v_out_datos_res)

f_res = []
v_in_res = []
v_out_res = []

for i in range(len(data_res)):
    if v_out_datos_res[i] < 5:
        f_res.append(f_datos_res[i])
        v_in_res.append(v_in_datos_res[i])
        v_out_res.append(v_out_datos_res[i])

f_res = np.array(f_res)
v_in_res = np.array(v_in_res)
v_out_res = np.array(v_out_res)

t_res = v_out_res/v_in_res

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(f_res, t_res, '.')
plt.grid(True)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('ransferencia')
plt.legend()
plt.show()

data_anti = pd.read_csv('Antiresonancia.csv')

f_datos_anti = data_anti.iloc[:,0]
f_datos_anti = pd.to_numeric(f_datos_anti)
v_in_datos_anti = data_anti.iloc[:,1]
v_in_datos_anti = pd.to_numeric(v_in_datos_anti)
v_out_datos_anti = data_anti.iloc[:,2]
v_out_datos_anti = pd.to_numeric(v_out_datos_anti)

f_anti = []
v_in_anti = []
v_out_anti = []

for i in range(len(data_anti)):
    if v_out_datos_anti[i] < 5:
        f_anti.append(f_datos_anti[i])
        v_in_anti.append(v_in_datos_anti[i])
        v_out_anti.append(v_out_datos_anti[i])

f_anti = np.array(f_anti)
v_in_anti = np.array(v_in_anti)
v_out_anti = np.array(v_out_anti)

t_anti = v_out_anti/v_in_anti

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(f_anti, t_anti, '.')
plt.grid(True)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Transferencia')
plt.show()

f_campana = f_res[79:179]
t_campana = t_res[79:179]

R_2 = 10000
w_s_0 = 2*np.pi*f_res[126]
w_p_0 = 2*np.pi*f_res[128]
w_m_0 = 2*np.pi*f_res[124]
Q_0 = w_s_0/(w_p_0 - w_m_0)
T_0 = max(t_res)
R_0 = (R_2/T_0) - R_2
L_0 = (Q_0*R_2)/(w_s_0*T_0)
C_0 = 1/(L_0*w_s_0**2)

def func_trans_res(f, R, L, C):
    return R_2/(np.sqrt((R_2 + R)**2 + (2*np.pi*f*L - 1/(2*np.pi*f*C))**2))

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(f_campana, func_trans_res(f_campana, R_0, L_0, C_0))
plt.grid(True)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Transferencia')
plt.show()
    
params_1, cov_1 = curve_fit(lambda f, R: func_trans_res(f, R, L_0, C_0), f_campana, t_campana, [R_0], bounds=(0,np.inf))

print(params_1)

params_2, cov_2 = curve_fit(lambda f, L, C: func_trans_res(f, params_1, L, C), f_campana, t_campana, [L_0, C_0], bounds=(0, np.inf))

print(params_2)

def func_trans_res_v2(f, T, w_s, Q):
    return R_2/(np.sqrt((R_2/T)**2 + ((2*np.pi*f*Q*R_2)/(w_s*T)-(Q*R_2*w_s)/(T*2*np.pi*f))**2))

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(f_campana, func_trans_res_v2(f_campana, T_0, w_s_0, Q_0))
plt.grid(True)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Transferencia')
plt.show()

params_3, cov_3 = curve_fit(func_trans_res_v2, f_campana, t_campana, [T_0, w_s_0, Q_0], bounds=(0, np.Inf))

print(params_3)
print(T_0, w_s_0, Q_0)

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(f_campana, t_campana, '.', label='T(f)')
plt.plot(f_campana, func_trans_res_v2(f_campana, params_3[0], params_3[1], params_3[2]), label='Ajuste')
plt.grid(True)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Transferencia')
plt.legend()
plt.show()
