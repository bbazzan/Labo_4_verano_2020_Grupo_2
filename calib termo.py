# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 23:17:08 2020

@author: bruno
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure 

data_f_c = pd.read_csv('frio_a_caliente')
data_n_r_tc = pd.read_csv('nitrogeno_r_tc')
data_n_r = pd.read_csv('nitrogeno_r')

data_f_c.drop(['$ Medicion  '], axis=1, inplace=True)
data_n_r_tc.drop(['$ Medicion  ', '  TiempoTC  ', '  VoltajeTC '], axis=1, inplace=True)
data_n_r.drop(['$ Medicion  ', '  TiempoTC  ', '  VoltajeTC ', '  TiempoLM  ', '  VoltajeLM  '], axis=1, inplace=True)

data_f_c.columns = ['Tiempo R', 'R', 'Tiempo TC', 'V TC', 'Tiempo LM', 'V LM']
data_n_r_tc.columns = ['Tiempo R', 'R', 'Tiempo TC', 'V TC']
data_n_r.columns = ['Tiempo R', 'R']

err_r_f_c = np.full(100, 0.8)
err_tc_f_c = np.full(100, 2.2)
err_lm_f_c = np.full(100, 0.75)
err_r_n_r_tc = np.full(20, 0.8)
err_tc_n_r_tc = np.full(20, 2.2)
err_r_n_r = np.full(260, 0.8)

tiempoR_f_c = data_f_c.iloc[:,0].to_numpy()
R_f_c = data_f_c.iloc[:,1].to_numpy()
tiempoTC_f_c = data_f_c.iloc[:,2].to_numpy()
vTC_f_c = data_f_c.iloc[:,3].to_numpy()
tiempoLM_f_c = data_f_c.iloc[:,4].to_numpy()
vLM_f_c = data_f_c.iloc[:,5].to_numpy()

tempR_f_c = np.zeros(100)
tempTC_f_c = np.zeros(100)
tempLM_f_c = np.zeros(100)

for i in range(100):
    tempR_f_c[i] = (-0.0039083 + np.sqrt(0.0039083**2 - 4*(-0.0000005775)*(1-(R_f_c[i]/100))))/(2*(-0.0000005775))
    tempLM_f_c[i] = vLM_f_c[i]/0.01

for i in range(8):
    tempTC_f_c[i] = 1000*(25.173462*vTC_f_c[i] - 1.1662878*vTC_f_c[i]**2 - 1.10833638*vTC_f_c[i]**3 - 0.89773540*vTC_f_c[i]**4 - 0.37342377*vTC_f_c[i]**5 - 0.086632643*vTC_f_c[i]**6 - 0.010450598*vTC_f_c[i]**7 - 0.00051920577*vTC_f_c[i]**8)

for i in range(8,100):
    tempTC_f_c[i] = 1000*(25.08355*vTC_f_c[i] + 0.07860106*vTC_f_c[i]**2 - 0.2503131*vTC_f_c[i]**3 + 0.08315270*vTC_f_c[i]**4 - 0.01228034*vTC_f_c[i]**5 + 0.0009804036*vTC_f_c[i]**6 - 0.00004413030*vTC_f_c[i]**7 + 0.000001057735*vTC_f_c[i]**8 - 0.00000001052755*vTC_f_c[i]**9)

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.errorbar(tiempoR_f_c, tempR_f_c, err_r_f_c, None, 'x')
plt.errorbar(tiempoTC_f_c, tempTC_f_c, err_tc_f_c, None, 'o')
plt.errorbar(tiempoLM_f_c, tempLM_f_c, err_lm_f_c, None, 'd')
plt.xlabel('Tiempo (s)')
plt.ylabel('Temperatura (°C)')
plt.grid(b=True)
plt.gca().legend(('R','TC','LM'))
plt.show()

tiempoR_n_r_tc = data_n_r_tc.iloc[:,0].to_numpy()
R_n_r_tc = data_n_r_tc.iloc[:,1].to_numpy()
tiempoTC_n_r_tc = data_n_r_tc.iloc[:,2].to_numpy()
vTC_n_r_tc = data_n_r_tc.iloc[:,3].to_numpy()

tempR_n_r_tc = np.zeros(20)
tempTC_n_r_tc = np.zeros(20)
tempLM_n_r_tc = np.zeros(20)

for i in range(20):
    tempTC_n_r_tc[i] = 1000*(25.173462*vTC_n_r_tc[i] - 1.1662878*vTC_n_r_tc[i]**2 - 1.10833638*vTC_n_r_tc[i]**3 - 0.89773540*vTC_n_r_tc[i]**4 - 0.37342377*vTC_n_r_tc[i]**5 - 0.086632643*vTC_n_r_tc[i]**6 - 0.010450598*vTC_n_r_tc[i]**7 - 0.00051920577*vTC_n_r_tc[i]**8)

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.errorbar(tiempoTC_n_r_tc, tempTC_n_r_tc, err_tc_n_r_tc, None, 'o')
plt.xlabel('Tiempo (s)')
plt.ylabel('Temperatura (°C)')
plt.grid(b=True)
plt.gca().legend('TC')
plt.show()

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.errorbar(tiempoR_n_r_tc, R_n_r_tc, err_r_n_r_tc, None, 'x')
plt.xlabel('Tiempo (s)')
plt.ylabel('Resistencia (Ohm)')
plt.grid(b=True)
plt.gca().legend('R')
plt.show()

tiempoR_n_r = data_n_r.iloc[:,0].to_numpy()
R_n_r = data_n_r.iloc[:,1].to_numpy()
    
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.errorbar(tiempoR_n_r, R_n_r, err_r_n_r, None, 'x')
plt.xlabel('Tiempo (s)')
plt.ylabel('Resistencia (Ohm)')
plt.grid(b=True)
plt.gca().legend('R')
plt.show()