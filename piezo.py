# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 08:00:21 2020

@author: Bruno
"""
#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from scipy.optimize import curve_fit
#from scipy import special
plt.rcParams.update({'font.size': 15})

#%% Osciloscopio
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
plt.yscale('log')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Transferencia')
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

'''
def func_trans_res(f, R, L, C):
    return R_2/(np.sqrt((R_2 + R)**2 + (2*np.pi*f*L - 1/(2*np.pi*f*C))**2))

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(f_campana, func_trans_res(f_campana, R_0, L_0, C_0))
plt.grid(True)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Transferencia')
plt.show()
'''    
#params_1, cov_1 = curve_fit(lambda f, R: func_trans_res(f, R, L_0, C_0), f_campana, t_campana, [R_0], bounds=(0,np.inf))

#print(params_1)

#params_2, cov_2 = curve_fit(lambda f, L, C: func_trans_res(f, params_1, L, C), f_campana, t_campana, [L_0, C_0], bounds=(0, np.inf))

#print(params_2)

def func_trans_res_v2(f, T, w_s, Q):
    return R_2/(np.sqrt((R_2/T)**2 + ((2*np.pi*f*Q*R_2)/(w_s*T)-(Q*R_2*w_s)/(T*2*np.pi*f))**2))

'''
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(f_campana, func_trans_res_v2(f_campana, T_0, w_s_0, Q_0))
plt.grid(True)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Transferencia')
plt.show()
'''
params_3, cov_3 = curve_fit(func_trans_res_v2, f_campana, t_campana, [T_0, w_s_0, Q_0], bounds=(0, np.Inf))

print('Parametros: ', params_3)
print('Estimaciones: ', T_0, w_s_0, Q_0)

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(f_campana, t_campana, '.', label='T(f)')
plt.plot(f_campana, func_trans_res_v2(f_campana, params_3[0], params_3[1], params_3[2]), label='Ajuste')
plt.grid(True)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Transferencia')
plt.legend()
plt.show()

#%% Lock-In Resonancia
camp_res = pd.read_csv('Campana resonancia fina(1)', header=None)
camp_res = camp_res.transpose()
camp_res.columns = ['Frecuencia', 'V_out', 'Fase']
err_res = pd.read_csv('Errores_resonancia_fina(2).csv', header=None)
err_res.columns = ['Error frecuencia', 'Error V_out', 'Error transferencia']
f = camp_res.iloc[:,0].to_numpy()
v_out = camp_res.iloc[:,2].to_numpy()
phi = camp_res.iloc[:,1].to_numpy()
v_in = np.zeros(len(v_out))
for i in range(len(v_out)):
    v_in[i] = 1/np.sqrt(2)
Trans = v_out/v_in

R_2_2 = 10000
w_s_0_2 = 2*np.pi*f[4752]
w_p_0_2 = 2*np.pi*f[4877]
w_m_0_2 = 2*np.pi*f[4625]
Q_0_2 = w_s_0_2/(w_p_0_2 - w_m_0_2)
T_0_2 = max(Trans)
R_0_2 = (R_2_2/T_0_2) - R_2_2
L_0_2 = (Q_0_2*R_2_2)/(w_s_0_2*T_0_2)
C_0_2 = 1/(L_0_2*w_s_0_2**2)

params_4, cov_4 = curve_fit(func_trans_res_v2, f, Trans, [T_0_2, w_s_0_2, Q_0_2], sigma=err_res.iloc[:,2], absolute_sigma=True)
print('Parametros: ', params_4)
print('Estimaciones: ', T_0_2, w_s_0_2, Q_0_2)

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.errorbar(f, Trans, err_res.iloc[:,2], err_res.iloc[:,0], '.', ecolor='r', errorevery=500, zorder=0, label='T(f)')
plt.plot(f, func_trans_res_v2(f, params_4[0], params_4[1], params_4[2]), 'tab:orange', zorder=10, label='Ajuste')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Transferencia')
plt.grid(True)
plt.legend()
plt.show()

#%% Lock-In Antiresonancia
camp_anti = pd.read_csv('ANTI FINA', header=None)
camp_anti = camp_anti.transpose()
camp_anti.columns = ['Frecuencia', 'V_out', 'Fase']

f_anti_fina = camp_anti.iloc[:,0].to_numpy()
v_out_anti_fina = camp_anti.iloc[:,1].to_numpy()
phi_anti_fina = camp_anti.iloc[:,2].to_numpy()
v_in_anti_fina = np.zeros(len(v_out_anti_fina))
for i in range(len(v_out_anti_fina)):
    v_in_anti_fina[i] = 1/np.sqrt(2)
Trans_anti_fina = v_out_anti_fina/v_in_anti_fina

err_anti = pd.read_csv('Errores_antiresonancia(2).csv', header=None)
err_anti.columns = ['Error frecuencia', 'Error V_out', 'Error transferencia']

L_calculada = (params_4[2]*R_2_2)/(params_4[1]*params_4[0])
C_calculada = 1/(L_calculada*params_4[1]**2)
C_2_0 = 1/(((2*np.pi*f_anti_fina[2116])**2)*L_calculada - (1/C_calculada))
w_res = params_4[1]
w_anti = 2*np.pi*f_anti_fina[2116]
capacitancia_2 = 1/((w_anti**2 - w_res**2)*L_calculada)

#def func_trans_anti(f_a, C_2):
    #return R_2_2/(R_2_2 + (2*np.pi*f_a*L_calculada - 1/(2*np.pi*f_a*C_calculada))/(1 - 2*np.pi*f_a*C_2*(2*np.pi*f_a*L_calculada - 1/(2*np.pi*f_a*C_calculada))))

def func_trans_anti(w, R, L, C):
    return R_2/(R_2 + np.sqrt(((w*L-1/(w*C))**2)/((1-w*capacitancia_2*(w*L-1/(w*C)))**2)))

#params_5, cov_5 = curve_fit(func_trans_anti, f_anti_fina, Trans_anti_fina, [capacitancia_2], sigma=err_anti.iloc[:,2], absolute_sigma=True)

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.errorbar(f_anti_fina, Trans_anti_fina, err_anti.iloc[:,2], err_anti.iloc[:,0], '.', ecolor='r', errorevery=250, label='T(f)')
plt.plot(f_anti_fina, func_trans_anti(f_anti_fina*2*np.pi, 3332, L_calculada, C_calculada), 'tab:orange', label='Ajuste')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Transferencia')
plt.grid(True)
#plt.yscale('log')
plt.legend()
plt.show()

#%% Lock-In v_in(f)
datos_v_in = pd.read_csv('v_in func f', header=None)
datos_v_in = datos_v_in.transpose()
datos_v_in.columns = ['Frecuencia', 'V_in', 'Fase']

f_v_in = datos_v_in.iloc[:,0]
v_in_v_in = datos_v_in.iloc[:,1]
phi_v_in = datos_v_in.iloc[:,2]

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(f_v_in, v_in_v_in, '.',)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('V_in (V)')
plt.grid(True)
plt.xlim(50095.75, 50098.25)
plt.show()

#%% Toda la señal

datos_todos = pd.read_csv('piezo_lock_in_grueso.csv', header=None)
datos_todos = datos_todos.transpose()
datos_todos.columns = ['Frecuencia', 'V_out', 'Fase']

w_entera = datos_todos.iloc[:,0]*2*np.pi

def func_entera(w, R, L, C):
    return R_2/(R_2 + np.sqrt(((R)**2+(w*L-1/(w*C))**2)/((w*capacitancia_2*R)**2+(1-w*capacitancia_2*(w*L-1/(w*C)))**2)))

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(w_entera, func_entera(w_entera, 3332, L_calculada, C_calculada))
plt.yscale('log')
plt.show()

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(datos_todos.iloc[:,0], datos_todos.iloc[:,1]*np.sqrt(2), '.')
plt.plot(datos_todos.iloc[:,0], func_entera(datos_todos.iloc[:,0]*2*np.pi, 3332, L_calculada, C_calculada))
plt.yscale('log')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Transferencia')
plt.grid(True)

#%% Reloj
datos_reloj_t_amb = pd.read_csv('reso piezo reloj', header=None)
datos_reloj_t_amb = datos_reloj_t_amb.transpose()
datos_reloj_t_amb.columns = ['Frecuencia', 'V_out', 'Fase']

err_t_amb = pd.read_csv('Errores_relojTA(2).csv', header=None)
err_t_amb.columns = ['Error frecuencia', 'Error V_out', 'Error transferencia']

f_reloj_t_amb = datos_reloj_t_amb.iloc[3:120,0]
v_out_reloj_t_amb = datos_reloj_t_amb.iloc[3:120,1]
phi_reloj_t_amb = datos_reloj_t_amb.iloc[3:120,2]
v_in_reloj_t_amb = np.zeros(len(v_out_reloj_t_amb))
for i in range(len(v_in_reloj_t_amb)):
    v_in_reloj_t_amb[i] = 0.5/np.sqrt(2)
trans_reloj_t_amb = v_out_reloj_t_amb/v_in_reloj_t_amb

datos_reloj_medioA = pd.read_csv('campana resonancia peltier 0,5A', header=None)
datos_reloj_medioA = datos_reloj_medioA.transpose()
datos_reloj_medioA.columns = ['Frecuencia', 'V_out', 'Fase']

err_medioA = pd.read_csv('Errores_reloj_medio_frio(2).csv', header=None)
err_medioA.columns = ['Error frecuencia', 'Error V_out', 'Error transferencia']

f_reloj_medioA = datos_reloj_medioA.iloc[3:120,0]
v_out_reloj_medioA = datos_reloj_medioA.iloc[3:120,1]
phi_reloj_medioA = datos_reloj_medioA.iloc[3:120,2]
v_in_reloj_medioA = np.zeros(len(v_out_reloj_medioA))
for i in range(len(v_in_reloj_medioA)):
    v_in_reloj_medioA[i] = 0.5/np.sqrt(2)
trans_reloj_medioA = v_out_reloj_medioA/v_in_reloj_medioA

datos_reloj_1A = pd.read_csv('campana resonancia peltier 1A', header=None)
datos_reloj_1A = datos_reloj_1A.transpose()
datos_reloj_1A.columns = ['Frecuencia', 'V_out', 'Fase']

err_1A = pd.read_csv('Errores_reloj_1_frio(2).csv', header=None)
err_1A.columns = ['Error frecuencia', 'Error V_out', 'Error transferencia']

f_reloj_1A = datos_reloj_1A.iloc[3:120,0]
v_out_reloj_1A = datos_reloj_1A.iloc[3:120,1]
phi_reloj_1A = datos_reloj_1A.iloc[3:120,2]
v_in_reloj_1A = np.zeros(len(v_out_reloj_1A))
for i in range(len(v_in_reloj_1A)):
    v_in_reloj_1A[i] = 0.5/np.sqrt(2)
trans_reloj_1A = v_out_reloj_1A/v_in_reloj_1A

datos_reloj_2A = pd.read_csv('campana resonancia peltier 2A', header=None)
datos_reloj_2A = datos_reloj_2A.transpose()
datos_reloj_2A.columns = ['Frecuencia', 'V_out', 'Fase']

err_2A = pd.read_csv('Errores_reloj_2_frio(2).csv', header=None)
err_2A.columns = ['Error frecuencia', 'Error V_out', 'Error transferencia']

f_reloj_2A = datos_reloj_2A.iloc[3:120,0]
v_out_reloj_2A = datos_reloj_2A.iloc[3:120,1]
phi_reloj_2A = datos_reloj_2A.iloc[3:120,2]
v_in_reloj_2A = np.zeros(len(v_out_reloj_2A))
for i in range(len(v_in_reloj_2A)):
    v_in_reloj_2A[i] = 0.5/np.sqrt(2)
trans_reloj_2A = v_out_reloj_2A/v_in_reloj_2A

datos_reloj_3A = pd.read_csv('campana resonancia peltier 3A', header=None)
datos_reloj_3A = datos_reloj_3A.transpose()
datos_reloj_3A.columns = ['Frecuencia', 'V_out', 'Fase']

err_3A = pd.read_csv('Errores_reloj_3_frio(2).csv', header=None)
err_3A.columns = ['Error frecuencia', 'Error V_out', 'Error transferencia']

f_reloj_3A = datos_reloj_3A.iloc[3:120,0]
v_out_reloj_3A = datos_reloj_3A.iloc[3:120,1]
phi_reloj_3A = datos_reloj_3A.iloc[3:120,2]
v_in_reloj_3A = np.zeros(len(v_out_reloj_3A))
for i in range(len(v_in_reloj_3A)):
    v_in_reloj_3A[i] = 0.5/np.sqrt(2)
trans_reloj_3A = v_out_reloj_3A/v_in_reloj_3A

datos_reloj_medioA_caliente = pd.read_csv('campana resonancia peltier caliente 0,5A', header=None)
datos_reloj_medioA_caliente = datos_reloj_medioA_caliente.transpose()
datos_reloj_medioA_caliente.columns = ['Frecuencia', 'V_out', 'Fase']

err_medioA_caliente = pd.read_csv('Errores_reloj_medio_caliente(2).csv', header=None)
err_medioA_caliente.columns = ['Error frecuencia', 'Error V_out', 'Error transferencia']

f_reloj_medioA_caliente = datos_reloj_medioA_caliente.iloc[3:120,0]
v_out_reloj_medioA_caliente = datos_reloj_medioA_caliente.iloc[3:120,1]
phi_reloj_medioA_caliente = datos_reloj_medioA_caliente.iloc[3:120,2]
v_in_reloj_medioA_caliente = np.zeros(len(v_out_reloj_medioA_caliente))
for i in range(len(v_in_reloj_medioA_caliente)):
    v_in_reloj_medioA_caliente[i] = 0.5/np.sqrt(2)
trans_reloj_medioA_caliente = v_out_reloj_medioA_caliente/v_in_reloj_medioA_caliente

datos_reloj_1A_caliente = pd.read_csv('campana resonancia peltier caliente 1A', header=None)
datos_reloj_1A_caliente = datos_reloj_1A_caliente.transpose()
datos_reloj_1A_caliente.columns = ['Frecuencia', 'V_out', 'Fase']

err_1A_caliente = pd.read_csv('Errores_reloj_1_caliente(2).csv', header=None)
err_1A_caliente.columns = ['Error frecuencia', 'Error V_out', 'Error transferencia']

f_reloj_1A_caliente = datos_reloj_1A_caliente.iloc[3:120,0]
v_out_reloj_1A_caliente = datos_reloj_1A_caliente.iloc[3:120,1]
phi_reloj_1A_caliente = datos_reloj_1A_caliente.iloc[3:120,2]
v_in_reloj_1A_caliente = np.zeros(len(v_out_reloj_1A_caliente))
for i in range(len(v_in_reloj_1A_caliente)):
    v_in_reloj_1A_caliente[i] = 0.5/np.sqrt(2)
trans_reloj_1A_caliente = v_out_reloj_1A_caliente/v_in_reloj_1A_caliente

#datos_reloj_resistencia = pd.read_csv('campana calentado con resistencia', header=None)
#datos_reloj_resistencia = datos_reloj_resistencia.transpose()
#datos_reloj_resistencia.columns = ['Frecuencia', 'V_out', 'Fase']

#f_reloj_resistencia = datos_reloj_resistencia.iloc[3:120,0]
#v_out_reloj_resistencia = datos_reloj_resistencia.iloc[3:120,1]
#phi_reloj_resistencia = datos_reloj_resistencia.iloc[3:120,2]
#v_in_reloj_resistencia = np.zeros(len(v_out_reloj_resistencia))
#for i in range(len(v_in_reloj_resistencia)):
#    v_in_reloj_resistencia[i] = 0.5/np.sqrt(2)
#trans_reloj_resistencia = v_out_reloj_resistencia/v_in_reloj_resistencia

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
#plt.plot(f_reloj_resistencia, trans_reloj_resistencia, '.', label='85,91°C')
plt.errorbar(f_reloj_1A_caliente, trans_reloj_1A_caliente, err_1A_caliente.iloc[3:120,2], err_1A_caliente.iloc[3:120,0], label='60,46°C')
plt.errorbar(f_reloj_medioA_caliente, trans_reloj_medioA_caliente, err_medioA_caliente.iloc[3:120,2], err_medioA_caliente.iloc[3:120,0], label='45,27°C')
plt.errorbar(f_reloj_t_amb, trans_reloj_t_amb, err_t_amb.iloc[3:120,2], err_t_amb.iloc[3:120,0], label='30,12°C')
plt.errorbar(f_reloj_medioA, trans_reloj_medioA, err_medioA.iloc[3:120,2], err_medioA.iloc[3:120,0], label='19,88°C')
plt.errorbar(f_reloj_1A, trans_reloj_1A, err_1A.iloc[3:120,2], err_1A.iloc[3:120,0], label='15,39°C')
plt.errorbar(f_reloj_2A, trans_reloj_2A, err_2A.iloc[3:120,2], err_2A.iloc[3:120,0], label='10,97°C')
plt.errorbar(f_reloj_3A, trans_reloj_3A, err_3A.iloc[3:120,2], err_3A.iloc[3:120,0], label='7,09°C')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Tranferencia')
plt.grid(True)
plt.legend()


#%% Resonancias
#for i,j in enumerate(trans_reloj_resistencia):
#    if j == max(trans_reloj_resistencia):
#        index_resistencia = i
for i,j in enumerate(trans_reloj_1A_caliente):
    if j == max(trans_reloj_1A_caliente):
        index_1A_caliente = i
for i,j in enumerate(trans_reloj_medioA_caliente):
    if j == max(trans_reloj_medioA_caliente):
        index_medioA_caliente = i
for i,j in enumerate(trans_reloj_t_amb):
    if j == max(trans_reloj_t_amb):
        index_t_amb = i
for i,j in enumerate(trans_reloj_medioA):
    if j == max(trans_reloj_medioA):
        index_medioA = i
for i,j in enumerate(trans_reloj_1A):
    if j == max(trans_reloj_1A):
        index_1A = i
for i,j in enumerate(trans_reloj_2A):
    if j == max(trans_reloj_2A):
        index_2A = i
for i,j in enumerate(trans_reloj_3A):
    if j == max(trans_reloj_3A):
        index_3A = i

#w_s_resistencia = f_reloj_resistencia[index_resistencia]*2*np.pi
w_s_1A_caliente = f_reloj_1A_caliente[index_1A_caliente]*2*np.pi
w_s_medioA_caliente = f_reloj_medioA_caliente[index_medioA_caliente]*2*np.pi
w_s_t_amb = f_reloj_t_amb[index_t_amb]*2*np.pi
w_s_medioA = f_reloj_medioA[index_medioA]*2*np.pi
w_s_1A = f_reloj_1A[index_1A]*2*np.pi
w_s_2A = f_reloj_2A[index_2A]*2*np.pi
w_s_3A = f_reloj_3A[index_3A]*2*np.pi

resonancias = [w_s_1A_caliente, w_s_medioA_caliente, w_s_t_amb, w_s_medioA, w_s_1A, w_s_2A, w_s_3A]
temperaturas = [60.46, 45.27, 30.12, 19.88, 15.39, 10.97, 7.09]

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.errorbar(temperaturas, resonancias, [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001], None, '.', label='w_s(T)')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Frecuencia (Hz)')
plt.grid(True)
plt.legend()
plt.show()

#%% Parametros
R_dos = 30000

w_m_1A_caliente = f_reloj_1A_caliente[47]*2*np.pi
w_p_1A_caliente = f_reloj_1A_caliente[65]*2*np.pi
T_1A_caliente = max(trans_reloj_1A_caliente)
Q_1A_caliente = w_s_1A_caliente/(w_p_1A_caliente - w_m_1A_caliente)
R_1A_caliente = (R_dos/T_1A_caliente) - R_dos
L_1A_caliente = (Q_1A_caliente*R_1A_caliente)/(w_s_1A_caliente*T_1A_caliente)
C_1A_caliente = 1/(L_1A_caliente*w_s_1A_caliente**2)

w_m_medioA_caliente = f_reloj_medioA_caliente[48]*2*np.pi
w_p_medioA_caliente = f_reloj_medioA_caliente[65]*2*np.pi
T_medioA_caliente = max(trans_reloj_medioA_caliente)
Q_medioA_caliente = w_s_medioA_caliente/(w_p_medioA_caliente - w_m_medioA_caliente)
R_medioA_caliente = (R_dos/T_medioA_caliente) - R_dos
L_medioA_caliente = (Q_medioA_caliente*R_medioA_caliente)/(w_s_medioA_caliente*T_medioA_caliente)
C_medioA_caliente = 1/(L_medioA_caliente*w_s_medioA_caliente**2)

w_m_t_amb = f_reloj_t_amb[44]*2*np.pi
w_p_t_amb = f_reloj_t_amb[60]*2*np.pi
T_t_amb = max(trans_reloj_t_amb)
Q_t_amb = w_s_t_amb/(w_p_t_amb - w_m_t_amb)
R_t_amb = (R_dos/T_t_amb) - R_dos
L_t_amb = (Q_t_amb*R_t_amb)/(w_s_t_amb*T_t_amb)
C_t_amb = 1/(L_t_amb*w_s_t_amb**2)

w_m_medioA = f_reloj_medioA[48]*2*np.pi
w_p_medioA = f_reloj_medioA[67]*2*np.pi
T_medioA = max(trans_reloj_medioA)
Q_medioA = w_s_medioA/(w_p_medioA - w_m_medioA)
R_medioA = (R_dos/T_medioA) - R_dos
L_medioA = (Q_medioA*R_medioA)/(w_s_medioA*T_medioA)
C_medioA = 1/(L_medioA*w_s_medioA**2)

w_m_1A = f_reloj_1A[46]*2*np.pi
w_p_1A = f_reloj_1A[65]*2*np.pi
T_1A = max(trans_reloj_1A)
Q_1A = w_s_1A/(w_p_1A - w_m_1A)
R_1A = (R_dos/T_1A) - R_dos
L_1A = (Q_1A*R_1A)/(w_s_1A*T_1A)
C_1A = 1/(L_1A*w_s_1A**2)

w_m_2A = f_reloj_2A[47]*2*np.pi
w_p_2A = f_reloj_2A[65]*2*np.pi
T_2A = max(trans_reloj_2A)
Q_2A = w_s_2A/(w_p_2A - w_m_2A)
R_2A = (R_dos/T_2A) - R_dos
L_2A = (Q_2A*R_2A)/(w_s_2A*T_2A)
C_2A = 1/(L_2A*w_s_2A**2)

w_m_3A = f_reloj_3A[51]*2*np.pi
w_p_3A = f_reloj_3A[66]*2*np.pi
T_3A = max(trans_reloj_3A)
Q_3A = w_s_3A/(w_p_3A - w_m_3A)
R_3A = (R_dos/T_3A) - R_dos
L_3A = (Q_3A*R_3A)/(w_s_3A*T_3A)
C_3A = 1/(L_3A*w_s_3A**2)

#%% Ajustes
f_reloj_t_amb = datos_reloj_t_amb.iloc[40:70,0]
v_out_reloj_t_amb = datos_reloj_t_amb.iloc[40:70,1]
phi_reloj_t_amb = datos_reloj_t_amb.iloc[40:70,2]
v_in_reloj_t_amb = np.zeros(len(v_out_reloj_t_amb))
for i in range(len(v_in_reloj_t_amb)):
    v_in_reloj_t_amb[i] = 0.5/np.sqrt(2)
trans_reloj_t_amb = v_out_reloj_t_amb/v_in_reloj_t_amb

f_reloj_medioA = datos_reloj_medioA.iloc[45:75,0]
v_out_reloj_medioA = datos_reloj_medioA.iloc[45:75,1]
phi_reloj_medioA = datos_reloj_medioA.iloc[45:75,2]
v_in_reloj_medioA = np.zeros(len(v_out_reloj_medioA))
for i in range(len(v_in_reloj_medioA)):
    v_in_reloj_medioA[i] = 0.5/np.sqrt(2)
trans_reloj_medioA = v_out_reloj_medioA/v_in_reloj_medioA

f_reloj_1A = datos_reloj_1A.iloc[45:73,0]
v_out_reloj_1A = datos_reloj_1A.iloc[45:73,1]
phi_reloj_1A = datos_reloj_1A.iloc[45:73,2]
v_in_reloj_1A = np.zeros(len(v_out_reloj_1A))
for i in range(len(v_in_reloj_1A)):
    v_in_reloj_1A[i] = 0.5/np.sqrt(2)
trans_reloj_1A = v_out_reloj_1A/v_in_reloj_1A

f_reloj_2A = datos_reloj_2A.iloc[45:75,0]
v_out_reloj_2A = datos_reloj_2A.iloc[45:75,1]
phi_reloj_2A = datos_reloj_2A.iloc[45:75,2]
v_in_reloj_2A = np.zeros(len(v_out_reloj_2A))
for i in range(len(v_in_reloj_2A)):
    v_in_reloj_2A[i] = 0.5/np.sqrt(2)
trans_reloj_2A = v_out_reloj_2A/v_in_reloj_2A

f_reloj_3A = datos_reloj_3A.iloc[45:75,0]
v_out_reloj_3A = datos_reloj_3A.iloc[45:75,1]
phi_reloj_3A = datos_reloj_3A.iloc[45:75,2]
v_in_reloj_3A = np.zeros(len(v_out_reloj_3A))
for i in range(len(v_in_reloj_3A)):
    v_in_reloj_3A[i] = 0.5/np.sqrt(2)
trans_reloj_3A = v_out_reloj_3A/v_in_reloj_3A

f_reloj_medioA_caliente = datos_reloj_medioA_caliente.iloc[45:75,0]
v_out_reloj_medioA_caliente = datos_reloj_medioA_caliente.iloc[45:75,1]
phi_reloj_medioA_caliente = datos_reloj_medioA_caliente.iloc[45:75,2]
v_in_reloj_medioA_caliente = np.zeros(len(v_out_reloj_medioA_caliente))
for i in range(len(v_in_reloj_medioA_caliente)):
    v_in_reloj_medioA_caliente[i] = 0.5/np.sqrt(2)
trans_reloj_medioA_caliente = v_out_reloj_medioA_caliente/v_in_reloj_medioA_caliente

f_reloj_1A_caliente = datos_reloj_1A_caliente.iloc[44:74,0]
v_out_reloj_1A_caliente = datos_reloj_1A_caliente.iloc[44:74,1]
phi_reloj_1A_caliente = datos_reloj_1A_caliente.iloc[44:74,2]
v_in_reloj_1A_caliente = np.zeros(len(v_out_reloj_1A_caliente))
for i in range(len(v_in_reloj_1A_caliente)):
    v_in_reloj_1A_caliente[i] = 0.5/np.sqrt(2)
trans_reloj_1A_caliente = v_out_reloj_1A_caliente/v_in_reloj_1A_caliente

frecuencias = [f_reloj_1A_caliente, f_reloj_medioA_caliente, f_reloj_t_amb, f_reloj_medioA, f_reloj_1A, f_reloj_2A, f_reloj_3A]
transferencias = [trans_reloj_1A_caliente, trans_reloj_medioA_caliente, trans_reloj_t_amb, trans_reloj_medioA, trans_reloj_1A, trans_reloj_2A, trans_reloj_3A]
Qs = [Q_1A_caliente, Q_medioA_caliente, Q_t_amb, Q_medioA, Q_1A, Q_2A, Q_3A]
Ts = [T_1A_caliente, T_medioA_caliente, T_t_amb, T_medioA, T_1A, T_2A, T_3A]
err_frecs = [err_1A_caliente.iloc[45:75,0], err_medioA_caliente.iloc[44:74,0], err_t_amb.iloc[40:70,0], err_medioA.iloc[45:75,0], err_1A.iloc[45:73,0], err_2A.iloc[45:75,0], err_3A.iloc[45:75,0]]
err_trans = [err_1A_caliente.iloc[45:75,2], err_medioA_caliente.iloc[44:74,2], err_t_amb.iloc[40:70,2], err_medioA.iloc[45:75,2], err_1A.iloc[45:73,2], err_2A.iloc[45:75,2], err_3A.iloc[45:75,2]]
params = []
params = []
cov = []

for i in range(7):
    params_i, cov_i = curve_fit(func_trans_res_v2, frecuencias[i], transferencias[i], [Ts[i], resonancias[i], Qs[i]])
    params.append(params_i)
    cov.append(cov_i)
    figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.errorbar(frecuencias[i], transferencias[i], err_trans[i], err_frecs[i],  '.')
    plt.plot(frecuencias[i], func_trans_res_v2(frecuencias[i], params_i[0], params_i[1], params_i[2]))
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Transferencia')
    plt.grid(True)
    plt.show()
    
temperaturas = [60.46, 45.27, 30.12, 19.88, 15.39, 10.97, 7.09]    
Qs = [Q_1A_caliente, Q_medioA_caliente, Q_t_amb, Q_medioA, Q_1A, Q_2A, Q_3A]
resonancias = [w_s_1A_caliente, w_s_medioA_caliente, w_s_t_amb, w_s_medioA, w_s_1A, w_s_2A, w_s_3A]

err_Q_1A_caliente = np.sqrt((0.01/(w_p_1A_caliente - w_m_1A_caliente))**2 + 2*((0.01*w_s_1A_caliente)/(w_p_1A_caliente - w_m_1A_caliente))**2)
err_Q_medioA_caliente =  np.sqrt((0.01/(w_p_medioA_caliente - w_m_medioA_caliente))**2 + 2*((0.01*w_s_medioA_caliente)/(w_p_medioA_caliente - w_m_medioA_caliente))**2)
err_Q_t_amb =  np.sqrt((0.01/(w_p_t_amb - w_m_t_amb))**2 + 2*((0.01*w_s_t_amb)/(w_p_t_amb - w_m_t_amb))**2)
err_Q_medioA =  np.sqrt((0.01/(w_p_medioA - w_m_medioA))**2 + 2*((0.01*w_s_medioA)/(w_p_medioA - w_m_medioA))**2)
err_Q_1A =  np.sqrt((0.01/(w_p_1A - w_m_1A))**2 + 2*((0.01*w_s_1A)/(w_p_1A - w_m_1A))**2)
err_Q_2A =  np.sqrt((0.01/(w_p_2A - w_m_2A))**2 + 2*((0.01*w_s_2A)/(w_p_2A - w_m_2A))**2)
err_Q_3A =  np.sqrt((0.01/(w_p_3A - w_m_3A))**2 + 2*((0.01*w_s_3A)/(w_p_3A - w_m_3A))**2)

errs_Qs = [err_Q_1A_caliente, err_Q_medioA_caliente, err_Q_t_amb, err_Q_medioA, err_Q_1A, err_Q_2A, err_Q_3A]

fig, ax1 = plt.subplots(figsize=(10,8))
ax1.set_xlabel('Temperatura (°C)')
ax1.set_ylabel('Frecuencia de resonancia (Hz)', color='tab:red')
ax1.errorbar(temperaturas, resonancias, [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], None, '.', color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.grid(True)
ax2 = ax1.twinx()
ax2.set_ylabel('Q', color='tab:blue')
ax2.errorbar(temperaturas, Qs, errs_Qs, None, 'd', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.grid(True)
fig.tight_layout()
plt.show()

#figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
#plt.errorbar(temperaturas, Ts, [0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003], None, '.')
#plt.xlabel('Temperatura (°C)')
#plt.ylabel('Amplitud')
#plt.grid(True)
#plt.show()