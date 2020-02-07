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

data.to_csv('datos_planchado.csv', index=False)

#%%Separo los distintos valores y tiempos correspondientes a cada termocupla
valores1 = data.iloc[225:650,1]
valores2 = data.iloc[225:650,3]
valores3 = data.iloc[225:650,5]
valores4 = data.iloc[225:650,7]
valores5 = data.iloc[225:650,9]
valores6 = data.iloc[225:650,11]
valores7 = data.iloc[225:650,13]

tiempos1 = data.iloc[225:650,0]
tiempos2 = data.iloc[225:650,2]
tiempos3 = data.iloc[225:650,4]
tiempos4 = data.iloc[225:650,6]
tiempos5 = data.iloc[225:650,8]
tiempos6 = data.iloc[225:650,10]
tiempos7 = data.iloc[225:650,12]

#%%Ploteo los datos para ver que onda
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(tiempos1, valores1, tiempos2, valores2, tiempos3, valores3, tiempos4, valores4, tiempos5, valores5, tiempos6, valores6, tiempos7, valores7)
plt.xlabel('Tiempo (s)')
plt.ylabel('Temperatura (°C)')
plt.show()

#%%Transpormo en arrays de numpy
valores1_np = valores1.to_numpy()
valores2_np = valores2.to_numpy()
valores3_np = valores3.to_numpy()
valores4_np = valores4.to_numpy()
valores5_np = valores5.to_numpy()
valores6_np = valores6.to_numpy()
valores7_np = valores7.to_numpy()

tiempos1_np = tiempos1.to_numpy()
tiempos2_np = tiempos2.to_numpy()
tiempos3_np = tiempos3.to_numpy()
tiempos4_np = tiempos4.to_numpy()
tiempos5_np = tiempos5.to_numpy()
tiempos6_np = tiempos6.to_numpy()
tiempos7_np = tiempos7.to_numpy()

#%%Para hacer el ciclo for pongo todo en dos matrices: una de valores y una de tiempos. Cada tira queda en una de las filas de la matriz
V = np.array([valores1_np, valores2_np, valores3_np, valores4_np, valores5_np, valores6_np, valores7_np])

T = np.array([tiempos1_np, tiempos2_np, tiempos3_np, tiempos4_np, tiempos5_np, tiempos6_np, tiempos7_np])


#%%Defino la funcion que uso para fittear y la pruebo con set de tiempos de prueba
x = 0.081
k = 0.000102
K = 380
T0 = 28.5

def func_difusividad(t, F0):
    
    return T0 + 2*(F0/K)*(np.sqrt(t*k/np.pi) * np.exp(-(x**2) / (4*k*t))- x/2 * special.erfc(x/(2*np.sqrt(k*t))))

t_prueba = np.linspace(2250, 6500, 100000)

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(t_prueba, func_difusividad(t_prueba,97812))
plt.xlabel('Tiempo (s)')
plt.ylabel('Temperatura (°C)')
plt.show()

#%%Ciclo for
p0 = 97812

p_opt = np.zeros(7)
p_cov = np.zeros(7)
F0 = np.zeros(7)
std = np.zeros(7)
TCs = ['TC 1', 'TC 2', 'TC 3', 'TC 4', 'TC 5', 'TC 6', 'TC 7']

for i in range(7):
    p_opt[i], p_cov[i] = curve_fit(func_difusividad, T[i], V[i], p0)
    F0[i] = p_opt[i]
    std[i] = np.sqrt(p_cov[i])
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(T[i], func_difusividad(T[i], F0[i]))
    plt.plot(T[i], V[i], 'kx')
    plt.title(TCs[i])
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Temperatura (°C)')
    plt.show()
    print('F0 = ', F0[i], '+/-', std[i])
