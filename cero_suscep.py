# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 20:13:32 2020

@author: bruno
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from scipy.optimize import curve_fit
from scipy import special

plt.rcParams.update({'font.size': 15})

cero = pd.read_csv('cero_csv.csv')
f_cero = cero.iloc[:,0]
f_cero = pd.to_numeric(f_cero)
x_cero = cero.iloc[:,1]
x_cero = pd.to_numeric(x_cero)
y_cero = cero.iloc[:,2]
y_cero = pd.to_numeric(y_cero)

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(f_cero, x_cero, '.', label='Re(V)')
plt.plot(f_cero, y_cero, '.', label='Im(V)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Voltaje (V)')
plt.grid(True)
plt.legend()