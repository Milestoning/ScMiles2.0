#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 10:37:37 2018

@author: Wei Wei


This subroutine takes the k and t, and calculate flux, probability,
free energy, committor, and MFPT.
"""

import os
import pandas as pd
import numpy as np
from network_check import pathway
#from voronoi_plot import voronoi_plot

__all__ = ['k_average','compute']


def find_ms_index(ms, ms_index):
    ms = sorted(ms)
    return(int(list(ms_index.keys())[list(ms_index.values()).index(ms)]))


def get_ms_of_cell(cell, ms_index):
    '''if reactant/product is a cell, return all milestones associated with this cell'''
    ms_of_cell = []
    for item in ms_index.values():
        if int(cell) in item:
            ms_of_cell.append(int(list(ms_index.keys())[list(ms_index.values()).index(item)]))
    return ms_of_cell


def k_average(k_count):
    '''convert count matrix to probability matrix'''
    dim = len(k_count)
    k_ave = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            if i != j:
                if np.sum(k_count[i, :]) != 0.0:
                    k_ave[i, j] = k_count[i, j] / np.sum(k_count[i, :]) 
    return k_ave


def k_error(k_count):
    '''return a new random k based on beta function'''
    dim = len(k_count)
    k = np.zeros((dim, dim))
    for i in range(dim):
        total = np.sum(k_count[i]) 
        for j in range(dim):
            a = k_count[i,j]
            if i == j or a == 0:
                continue
            if total == a:
                k[i, j] = 1.0
            b = total - a
            if b > 0: 
                k[i, j] = np.random.beta(a, b)
        if sum(k[i, :]) != 0:
            k[i, :] = k[i, :] / sum(k[i, :])
    return k

 
def t_error(t, t_std):
    '''return a new set of life time based on the std'''
    return np.abs(np.random.normal(t, t_std, len(t))).tolist()


def committor(parameter, k):
    '''committor function'''
    kk = k.copy()
    for i in parameter.reactant_milestone:
        kk[i] = [0 for j in k[i]]
    for i in parameter.product_milestone:
        kk[i] = [0 for j in k[i]]
        kk[i][i] = 1.0
    c = np.linalg.matrix_power(kk,1000000000)
    A = np.ones((len(c),1))
    return np.matmul(c,A)

    
def flux(k):
    '''flux calculation'''
    kk = k.copy()
    kk_trans = np.transpose(kk)
    e_v, e_f = np.linalg.eig(kk_trans)
    idx = np.abs(e_v - 1).argmin()  
    q = [i.real for i in e_f[:, idx]]
    q = np.array(q)
    if np.all(q < 0):
        q = -1 * q
    return q


def prob(q, t):
    '''probability calculation'''
    p = np.transpose(q) * np.squeeze(t)
    p_norm = p / np.sum(p)
    return p_norm


def free_energy(p):
    '''free energy from probability'''
    return -1.0 * np.log(p)
    

def get_boundary(parameter, ms_index):
    '''
    get the index number for reactant and product state

    If a K matrix is like this, reactant is 1_2, this function will return 0 as the first row/column indicates 1_2
    1_2 2_3 3_4
      0   1   0
    0.5   0 0.5
      0   1   0
    '''
    if len(parameter.reactant) == 2:
        bc1 = sorted(parameter.reactant)
        if bc1 in ms_index.values():
            parameter.reactant_milestone.append(int(list(ms_index.keys())[list(ms_index.values()).index(bc1)]))
        else:
            parameter.reactant_milestone.append(-1)
    else:
        for item in ms_index.values():
            if int(parameter.reactant[0]) in item and int(list(ms_index.keys())[list(ms_index.values()).index(item)]) \
                    not in parameter.reactant_milestone:
                parameter.reactant_milestone.append(int(list(ms_index.keys())[list(ms_index.values()).index(item)]))
            
    if len(parameter.product) == 2:
        bc2 = sorted(parameter.product)
        if bc2 in ms_index.values():   
            parameter.product_milestone.append(int(list(ms_index.keys())[list(ms_index.values()).index(bc2)]))
        else:
            parameter.product_milestone.append(-1)
    else:
        for item in ms_index.values():
            if int(parameter.product[0]) in item and int(list(ms_index.keys())[list(ms_index.values()).index(item)]) \
                    not in parameter.product_milestone:
                parameter.product_milestone.append(int(list(ms_index.keys())[list(ms_index.values()).index(item)]))            


def MFPT(parameter, kk, t):
    '''MFPT based on flux'''
    k = kk.copy()
    for i in parameter.product_milestone:
        k[i] = [0 for j in k[i]]
        for j in parameter.reactant_milestone:
            k[i][j] = 1.0 / len(parameter.reactant_milestone)   
    q = flux(k)
    qf = 0
    for i in parameter.product_milestone:
        qf += q[i]
    tau = np.dot(q, t) / qf
    return float(tau)
    

def MFPT2(parameter, k, t):
    '''MFPT based on inverse of K'''
    dim = len(k)
    I = np.identity(dim)
    k2 = k.copy()
    for i in parameter.product_milestone:
        if i == -1:
            return -1
        k2[i] = [0 for i in k2[i]]
    
    p0 = np.zeros(dim)
    for i in parameter.reactant_milestone:
        if i == -1:
            return -1
        p0[i] = 1 / (len(parameter.reactant_milestone))
        
    if np.linalg.det(np.mat(I) - np.mat(k2)) == 0.0:
        parameter.sing = True
        return -1
    else:
        parameter.sing = False
        tau = p0 * np.linalg.inv(I - np.mat(k2)) * np.transpose(np.mat(t))
    return float(tau)

def exit_time(parameter,k,t,t_matrix,committor):
    dim = len(k)
    I = np.identity(dim)
    k2 = k.copy()
    for i in parameter.product_milestone:
        if i == -1:
            return -1
        k2[i] = [0 for i in k2[i]]
        t[i] = 0
    for i in parameter.reactant_milestone:
        if i == -1:
            return -1
        k2[i] = [0 for i in k2[i]]
        t[i] = 0
    if np.linalg.det(np.mat(I) - np.mat(k2)) == 0.0:
        parameter.sing = True
        return -1
    else:
        parameter.sing = False
        #print(np.linalg.inv(I - np.mat(k2)))
        #print(np.transpose(np.mat(t)))
        #exit time
        tau = np.linalg.inv(I - np.mat(k2)) * np.transpose(np.mat(t))
        #directional_exit
        inverse_matrix = np.linalg.inv(I-np.mat(k2))
        e_vector_product = np.zeros((len(t),1))
        e_vector_product[parameter.product_milestone] = 1
        directional_exit = []
        
        for i in range(len(t)):
            if i in parameter.product_milestone or i in parameter.reactant_milestone:
                directional_exit.append(0)
                continue
            e_vector_i = np.zeros((len(t),1))       
            e_vector_i[i] = 1
            calc1 = np.matmul(np.transpose(e_vector_i), inverse_matrix)
            calc2 = np.matmul(calc1, t_matrix)
            calc3 = np.matmul(calc2, inverse_matrix)
            calc4 = np.matmul(calc3,e_vector_product)
            calc4[0] = calc4[0]/committor[i]
            directional_exit.append(float(calc4[0]))
    print(directional_exit)
    #return float(tau)
    tau_list = []
    for i in range(len(tau)):
        tau_list.append(float(tau[i][0]))
    #print(tau_list)
    return tau_list, 0

def compute(parameter):
    from math import sqrt
    path = parameter.currentPath
    filepath_t = path + '/life_time.txt'
    t_raw = pd.read_fwf(filepath_t, header=1).values
    t = (t_raw[0, :]).tolist()
    t_std = (t_raw[1, :]).tolist()
    dimension = len(t_raw[0])
    kc_raw = pd.read_fwf(path + '/k.txt', header=1).values
    kc = [[float(j) for j in i] for i in kc_raw[0:dimension,0:dimension].tolist()]
    k = k_average(np.array(kc))
    t_matrix = []
    count = 0
    with open(path + '/t.txt') as f:
        for line in f:
            count += 1
            if count <= 2:
                continue
            info = line.split()
            for i in range(len(info)):
                info[i] = float(info[i])
            t_matrix.append(info)
    print(t_matrix)
    t_matrix = np.array(t_matrix)
    print(t_matrix)
    ms_list = np.load(path + '/ms_index.npy', allow_pickle=True).item()
    
    kk = k.copy()
    tt = t.copy()
    parameter.reactant_milestone = []
    parameter.product_milestone = []
    get_boundary(parameter, ms_list)
    
    kk = k_average(np.array(kk))
    
    kk_cyc = k.copy()
    for i in parameter.product_milestone:
        kk_cyc[i] = [0 for j in k[i]]
        for j in parameter.reactant_milestone:
            kk_cyc[i][j] = 1.0 / len(parameter.reactant_milestone)   
    q_cyc = flux(kk_cyc)
    
    q = flux(kk)    
    p = prob(q,tt)
    energy = free_energy(p)
    tau1 = MFPT(parameter, kk, tt)
    tau2 = MFPT2(parameter, kk, tt)
    #print("exit time is")
    #print(exit_t)

    energy_samples = []
    MFPT_samples = []
    MFPT2_samples = []
    energy_err = []
    MFPT_err = []
    MFPT_err2 = []    
    
    for i in range(len(t_std)):
        t_std[i] = t_std[i]/sqrt(parameter.trajPerLaunch)
    
    for i in range(parameter.err_sampling):
        k_err = k_error(np.mat(kc))
        #if not isinstance(t_std,float):
            #t_std = tt
        t_err = t_error(tt, t_std)
        q_temp = flux(k_err)
        p_temp = prob(q_temp,tt)
        energy_samples.append(free_energy(p_temp))
        MFPT_er = MFPT(parameter, k_err, t_err)
        MFPT_samples.append(MFPT_er)
        MFPT_er2 = MFPT2(parameter, k_err, t_err)
        MFPT2_samples.append(MFPT_er2)
    
    
#    MFPT_samples = np.log10(MFPT_samples)
#    import matplotlib.pyplot as plt
#    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':200})
#    n, bins, patches = plt.hist(MFPT_samples, bins=50)
#    plt.show()
#    plt.savefig('MFPT_hist.png')
    
    for i in range(dimension):        
        energy_err.append(np.std(np.array(energy_samples)[:,i], ddof=1))
    MFPT_err = float(np.std(MFPT_samples, ddof=1))
    MFPT_err2 = float(np.std(MFPT2_samples, ddof=1))
        
    with open(path + '/results.txt', 'w+') as f1:
        print('{:>4} {:>4} {:>10} {:>8} {:>13} {:>10}'.format('a1', 'a2', 'q', 'p', 'freeE(kT)', 'freeE_err'),file=f1)
        keyIndex = 0
        for i in range(len(q)):
            while True:
                if keyIndex not in ms_list:
                    keyIndex += 1
                else:
                    break
            print('{:4d} {:4d} {:10.5f} {:8.5f} {:13.5f} {:10.5f}'.format(ms_list[keyIndex][0], ms_list[keyIndex][1], 
                  q[i], p[i], energy[i], energy_err[i]), file=f1)
            keyIndex += 1  
        print('\n\n',file=f1)
        print("MFPT is {:15.8e} fs, with an error of {:15.8e}, from eigenvalue method.".format(tau1, MFPT_err),file=f1)  
        print("MFPT is {:15.8e} fs, with an error of {:15.8e}, from inverse method.".format(tau2, MFPT_err2),file=f1) 
        
    c = committor(parameter, kk)
    exit_t, directional_exit_t = exit_time(parameter, kk, tt, t_matrix, c)

    m = []
    for ms in ms_list.values():
        m.append(ms)
    with open(path + '/committor.txt', 'w+') as f1:
        print('\n'.join([''.join(['{:>15}'.format(str(item)) for item in m])]),file=f1)
        print('\n'.join([''.join(['{:15.8f}'.format(item) for item in np.squeeze(c)])]),file=f1)
    with open(path + '/exit_time.txt', 'w+') as f1:
        print('\n'.join([''.join(['{:>15}'.format(str(item)) for item in m])]),file=f1)
        tmp = ''
        for i in range(len(exit_t)):
            tmp += ('\n'.join([''.join(['{:15.8f}'.format(exit_t[i])])]))
        print(tmp, file=f1)

    if parameter.data_file == True:    
        get_start_end_lifetime(parameter)  
    
    if not parameter.sing:
        parameter.MFPT = tau1
    else:
        parameter.MFPT = 0
        
#    voronoi_plot(parameter, m, c, energy, ms_list)
    index = []
    for i in range(len(ms_list)):
        index.append(ms_list[i])
    return k, index, q # q_cyc

def get_next_frame_num(path):
    next_frame = 1
    while True:
        if os.path.exists(path + '/' + str(next_frame)):
            next_frame += 1
        else:
            return next_frame

def get_start_end_lifetime(parameter):
    import pandas as pd
    import os
    import re
    data = []
    for ms in parameter.MS_list:    
        lst = re.findall('\d+',ms)
        name = lst[0] + '_' + lst[1]
        ms_path = parameter.crdPath + '/' + name + '/' + str(parameter.iteration)
        next_frame = get_next_frame_num(ms_path)
        for traj in range(1, next_frame):
            tmp = []
            traj_path = ms_path + '/' + str(traj) 
            if os.path.isfile(traj_path + '/start.txt'):
                start = pd.read_csv(traj_path + '/start.txt', header=None, delimiter=r'\s+').values.tolist()[0]
            else:
                start = ['N','N']
            if os.path.isfile(traj_path + '/end.txt'):
                end = pd.read_csv(traj_path + '/end.txt', header=None, delimiter=r'\s+').values.tolist()[0]
            else:
                end = ['N','N']
            if os.path.isfile(traj_path + '/lifetime.txt'):
                lifetime = pd.read_csv(traj_path + '/lifetime.txt', header=None, delimiter=r'\s+').values.tolist()[0]
            else:
                lifetime = 'N'
            if os.path.isfile(traj_path + '/enhanced'):
                enhanced = 'Enhanced'
            else:
                enhanced = 'NotEnhanced'
            tmp = [parameter.iteration, start[0], start[1], end[0], end[1], lifetime[0], enhanced, traj_path]
            '''            
            tmp.append(str(start[0]))
            tmp.append(str(start[1]))
            tmp.append(str(end[0]))
            tmp.append(str(end[1]))
            tmp.append(str(lifetime))
            '''
            data.append(tmp)
            #print(data)
    with open(parameter.currentPath + '/iteration_data.txt', 'w') as f1:
    	for item in data:
            f1.write(" ".join(map(str,item)) + '\n')
    if parameter.iteration == 1:
        with open(parameter.currentPath + '/all_data.txt', 'w') as f1:
            f1.write('Iteration start[0] start[1] end[0] end[1] Lifetime Enhancement Path')
            for item in data:
                f1.write(" ".join(map(str,item)) + '\n')
    else:
        with open(parameter.currentPath + '/all_data.txt', 'a') as f1:
            for item in data:
                f1.write(" ".join(map(str,item)) + '\n')

if __name__ == '__main__':
    from parameters import *
    new = parameters()
    new.initialize()
    new.iteration = 1
    compute(new)
    print('{:e}'.format(new.MFPT))
