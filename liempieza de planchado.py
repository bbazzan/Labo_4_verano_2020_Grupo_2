# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:08:13 2020

@author: Bruno
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from scipy.optimize import curve_fit
from scipy import special

#%%Importo los datos
data = pd.read_csv('04-February-2020_08-17-12_', sep=';')

data.columns = ['Canal 1', 'Tiempo 1', 'Valor 1', 'Canal 2', 'Tiempo 2', 'Valor 2', 'Canal 3', 'Tiempo 3', 'Valor 3', 'Canal 4', 'Tiempo 4', 'Valor 4', 'Canal 5', 'Tiempo 5', 'Valor 5', 'Canal 6', 'Tiempo 6', 'Valor 6', 'Canal 7', 'Tiempo 7', 'Valor 7']

data = data.drop(columns=['Canal 1', 'Canal 2', 'Canal 3', 'Canal 4', 'Canal 5', 'Canal 6' ,'Canal 7'])

#data.to_csv('datos_planchado.csv', index=False)

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
for i in [0, 2, 4, 6, 8, 10, 12]:
    plt.plot(data.iloc[:,i], data.iloc[:,i+1])
plt.xlabel('Tiempo (s)')
plt.ylabel('Temperatura (°C)')
plt.show()


#%%Separo los distintos valores y tiempos correspondientes a cada termocupla
valores1 = data.iloc[125:350,1]
valores2 = data.iloc[125:350,3]
valores3 = data.iloc[150:350,5]
valores4 = data.iloc[200:350,7]
valores5 = data.iloc[225:350,9]
valores6 = data.iloc[250:350,11]
valores7 = data.iloc[275:350,13]

tiempos1 = data.iloc[125:350,0]
tiempos2 = data.iloc[125:350,2]
tiempos3 = data.iloc[150:350,4]
tiempos4 = data.iloc[200:350,6]
tiempos5 = data.iloc[225:350,8]
tiempos6 = data.iloc[250:350,10]
tiempos7 = data.iloc[275:350,12]

#%%Transformo en arrays de numpy
valores1_np = valores1.to_numpy() 
valores2_np = valores2.to_numpy()
valores3_np = valores3.to_numpy()
valores4_np = valores4.to_numpy()
valores5_np = valores5.to_numpy()
valores6_np = valores6.to_numpy()
valores7_np = valores7.to_numpy()

v0_1 = valores1_np[0]
v0_2 = valores2_np[0]
v0_3 = valores3_np[0]
v0_4 = valores4_np[0]
v0_5 = valores5_np[0]
v0_6 = valores6_np[0]
v0_7 = valores7_np[0]

valores1_np = valores1_np - v0_1
valores2_np = valores2_np - v0_2
valores3_np = valores3_np - v0_3
valores4_np = valores4_np - v0_4
valores5_np = valores5_np - v0_5
valores6_np = valores6_np - v0_6
valores7_np = valores7_np - v0_7

tiempos1_np = tiempos1.to_numpy()
tiempos2_np = tiempos2.to_numpy()
tiempos3_np = tiempos3.to_numpy()
tiempos4_np = tiempos4.to_numpy()
tiempos5_np = tiempos5.to_numpy()
tiempos6_np = tiempos6.to_numpy()
tiempos7_np = tiempos7.to_numpy()

t0_1 = tiempos1_np[0]
t0_2 = tiempos2_np[0]
t0_3 = tiempos3_np[0]
t0_4 = tiempos4_np[0]
t0_5 = tiempos5_np[0]
t0_6 = tiempos6_np[0]
t0_7 = tiempos7_np[0]

tiempos1_np = tiempos1_np - t0_1 + 0.000001
tiempos2_np = tiempos2_np - t0_2 + 0.000001
tiempos3_np = tiempos3_np - t0_3 + 0.000001
tiempos4_np = tiempos4_np - t0_4 + 0.000001
tiempos5_np = tiempos5_np - t0_5 + 0.000001
tiempos6_np = tiempos6_np - t0_6 + 0.000001
tiempos7_np = tiempos7_np - t0_7 + 0.000001

#%%Ploteo los datos para ver que onda
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(tiempos1_np, valores1_np, tiempos2_np, valores2_np, tiempos3_np, valores3_np, tiempos4_np, valores4_np, tiempos5_np, valores5_np, tiempos7_np, valores7_np)
plt.xlabel('Tiempo (s)')
plt.ylabel('Temperatura (°C)')
plt.show()

#%%Para hacer el ciclo for pongo todo en dos matrices: una de valores y una de tiempos. Cada tira queda en una de las filas de la matriz
V = np.array([valores1_np, valores2_np, valores3_np, valores4_np, valores5_np, valores7_np])

T = np.array([tiempos1_np, tiempos2_np, tiempos3_np, tiempos4_np, tiempos5_np, tiempos7_np])


#%%Defino la funcion que uso para fittear
def temperatura(ts, x, A, B):
    
    return (A*((B*ts/np.pi)**0.5*np.exp(-(x**2)/(4*B*ts))-x/2*(1-special.erfc(x/(2*(B*ts)**0.5))))) 

#%%Ciclo for
'''
x = np.array([0.0814, 0.1231, 0.164, 0.2119, 0.2496, 0.4105])
TCs = ['TC 1', 'TC 2', 'TC 3', 'TC 4', 'TC 5', 'TC 7']
A = np.zeros(6)
B = np.zeros(6)
errA = np.zeros(6)
errB = np.zeros(6)

for i in range(6):
    p_opt, p_cov = curve_fit(lambda ts, A, B:temperatura(ts, x[i], A, B), T[i], V[i], [514.8, 0.0002])
    A[i] = p_opt[0]
    B[i] = p_opt[1]
    errA[i] = np.sqrt(p_cov[0,0])
    errB[i] = np.sqrt(p_cov[1,1])
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(T[i], temperatura(T[i], x[i], A[i], B[i]))
    plt.errorbar(T[i], V[i], err_temp, err_t, color='r')
    plt.title(TCs[i])
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Temperatura (°C)')
    plt.show()
    print(A[i])
    print(errA[i])
    print(B[i])
    print(errB[i])
'''

#%%
x  = np.array([0.0814, 0.1231, 0.164, 0.2119, 0.2496, 0.4105])
TCs = ['TC 1', 'TC 2', 'TC 3', 'TC 4', 'TC 5', 'TC 7']

p_opt_1, p_cov_1 = curve_fit(lambda ts, A, B:temperatura(ts, x[0], A, B), T[0], V[0], [514.8, 0.0002])
A_1 = p_opt_1[0]
B_1 = p_opt_1[1]
err_A_1 = np.sqrt(p_cov_1[0,0])
err_B_1 = np.sqrt(p_cov_1[1,1])
err_temp_1 = np.zeros(225)
err_temp_1[:] = 2.2
err_t_1 = np.zeros(225)
err_t_1[:] = 1
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(T[0], temperatura(T[0], x[0], A_1, B_1))
plt.errorbar(T[0], V[0], err_temp_1, err_t_1, color='r')
plt.title(TCs[0])
plt.xlabel('Tiempo (s)')
plt.ylabel('Temperatura (°C)')
print('F0 =', A_1)
print('err_F0 =', err_A_1)
print('k =', B_1)
print('err_k =', err_B_1)

p_opt_2, p_cov_2 = curve_fit(lambda ts, A, B:temperatura(ts, x[1], A, B), T[1], V[1], [514.8, 0.0002])
A_2 = p_opt_2[0]
B_2 = p_opt_2[1]
err_A_2 = np.sqrt(p_cov_1[0,0])
err_B_2 = np.sqrt(p_cov_1[1,1])
err_temp_2 = np.zeros(225)
err_temp_2[:] = 2.2
err_t_2 = np.zeros(225)
err_t_2[:] = 1
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(T[1], temperatura(T[1], x[1], A_2, B_2))
plt.errorbar(T[1], V[1], err_temp_2, err_t_2, color='r')
plt.title(TCs[1])
plt.xlabel('Tiempo (s)')
plt.ylabel('Temperatura (°C)')
print('F0 =', A_2)
print('err_F0 =', err_A_2)
print('k =', B_2)
print('err_k =', err_B_2)

p_opt_3, p_cov_3 = curve_fit(lambda ts, A, B:temperatura(ts, x[2], A, B), T[2], V[2], [514.8, 0.0002])
A_3 = p_opt_3[0]
B_3 = p_opt_3[1]
err_A_3 = np.sqrt(p_cov_3[0,0])
err_B_3 = np.sqrt(p_cov_3[1,1])
err_temp_3 = np.zeros(200)
err_temp_3[:] = 2.2
err_t_3 = np.zeros(200)
err_t_3[:] = 1
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(T[2], temperatura(T[2], x[2], A_3, B_3))
plt.errorbar(T[2], V[2], err_temp_3, err_t_3, color='r')
plt.title(TCs[2])
plt.xlabel('Tiempo (s)')
plt.ylabel('Temperatura (°C)')
print('F0 =', A_3)
print('err_F0 =', err_A_3)
print('k =', B_3)
print('err_k =', err_B_3)

p_opt_4, p_cov_4 = curve_fit(lambda ts, A, B:temperatura(ts, x[3], A, B), T[3], V[3], [514.8, 0.0002])
A_4 = p_opt_4[0]
B_4 = p_opt_4[1]
err_A_4 = np.sqrt(p_cov_4[0,0])
err_B_4 = np.sqrt(p_cov_4[1,1])
err_temp_4 = np.zeros(150)
err_temp_4[:] = 2.2
err_t_4 = np.zeros(150)
err_t_4[:] = 1
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(T[3], temperatura(T[3], x[3], A_4, B_4))
plt.errorbar(T[3], V[3], err_temp_4, err_t_4, color='r')
plt.title(TCs[3])
plt.xlabel('Tiempo (s)')
plt.ylabel('Temperatura (°C)')
print('F0 =', A_4)
print('err_F0 =', err_A_4)
print('k =', B_4)
print('err_k =', err_B_4)

p_opt_5, p_cov_5 = curve_fit(lambda ts, A, B:temperatura(ts, x[4], A, B), T[4], V[4], [514.8, 0.0002])
A_5 = p_opt_5[0]
B_5 = p_opt_5[1]
err_A_5 = np.sqrt(p_cov_5[0,0])
err_B_5 = np.sqrt(p_cov_5[1,1])
err_temp_5 = np.zeros(125)
err_temp_5[:] = 2.2
err_t_5 = np.zeros(125)
err_t_5[:] = 1
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(T[4], temperatura(T[4], x[4], A_5, B_5))
plt.errorbar(T[4], V[4], err_temp_5, err_t_5, color='r')
plt.title(TCs[4])
plt.xlabel('Tiempo (s)')
plt.ylabel('Temperatura (°C)')
print('F0 =', A_5)
print('err_F0 =', err_A_5)
print('k =', B_5)
print('err_k =', err_B_5)

p_opt_7, p_cov_7 = curve_fit(lambda ts, A, B:temperatura(ts, x[5], A, B), T[5], V[5], [514.8, 0.0002])
A_7 = p_opt_7[0]
B_7 = p_opt_7[1]
err_A_7 = np.sqrt(p_cov_7[0,0])
err_B_7 = np.sqrt(p_cov_7[1,1])
err_temp_7 = np.zeros(75)
err_temp_7[:] = 2.2
err_t_7 = np.zeros(75)
err_t_7[:] = 1
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(T[5], temperatura(T[5], x[5], A_7, B_7))
plt.errorbar(T[5], V[5], err_temp_7, err_t_7, color='r')
plt.title(TCs[5])
plt.xlabel('Tiempo (s)')
plt.ylabel('Temperatura (°C)')
print('F0 =', A_7)
print('err_F0 =', err_A_7)
print('k =', B_7)
print('err_k =', err_B_7)

plt.show()

