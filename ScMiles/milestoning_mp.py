#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 11:23:40 2018

@author: 
"""

from milestone import milestone
from compute import *
import numpy as np
from log import log
from datetime import datetime
import os
from milestones import milestones
from additional_functions import *
from math import sqrt

__all__ = ['milestoning']


def del_restarts(parameter, path, times, total_fs: int, freq: int) -> None:
    import os
    lst = []
    for time in times:
        lst.append(time // freq)
    for i in range(1, total_fs // 100 + 1):
        if i not in lst:
            name = path + '/' + parameter.outputname + '.' + str(i * freq) + '.coor'
            if os.path.isfile(name):
                os.remove(name)
            name = path + '/' + parameter.outputname + '.' + str(i * freq) + '.vel'
            if os.path.isfile(name):
                os.remove(name)
            name = path + '/' + parameter.outputname + '.' + str(i * freq) + '.xsc'
            if os.path.isfile(name):
                os.remove(name)


def K_order(k, t, t_std, index, t_matrix, parameter):
    """
    argv:
          k: original K matrix, random order 
              
          t: original t array, same order as K
          
          t_std: standard deviation of t, same order as K
          
          index: milestones index for original K and t
   
    return:
          ordered K, t, t_std, and the new index for them
    """
#    index_new = dict(sorted(index.items(), key=lambda x:x[1]))
    index_new = {}
    for i, ms in enumerate(sorted(index.values(), key=lambda x:x[0])):
        index_new[i] = ms
    mapping = []
    for i, ms in enumerate(list(index_new.values())):
        mapping.append([k for k,v in index.items() if v == ms][0])
    dimension = len(k)
    k_new = np.zeros((dimension, dimension)).astype(int)
    t_matrix_new = np.zeros((dimension, dimension)).astype(float)
    t_new = np.zeros(dimension)
    t_std_new = np.zeros(dimension)
    for dimY in range(dimension):
        t_new[dimY] = t[mapping[dimY],[0]]
        t_std_new[dimY] = t_std[mapping[dimY],[0]]
        for dimX in range(dimension):
            if k[mapping[dimX]][mapping[dimY]] <= parameter.k_cutoff:
                k_new[dimX][dimY]= 0
            else:
                k_new[dimX][dimY] = k[mapping[dimX]][mapping[dimY]]
            t_matrix_new[dimX][dimY] = t_matrix[mapping[dimX]][mapping[dimY]]
    return k_new, t_new, t_std_new, index_new, t_matrix_new


def backup(parameter, files: list) -> None:
    from shutil import copy
    import os
    time = str(datetime.now()).split('.')[0]
    for file in files:
        if os.path.isfile(parameter.currentPath + '/' + file): 
            backup_Folder = parameter.outputPath + '/' + str(parameter.iteration) + '_' + time
            create_folder(backup_Folder)
            copy(parameter.outputPath + '/current' + file, backup_Folder + file)
        
def get_traj_amount(struPath):
    '''
    Description: This function is used to find the next non-existent folder
        So if we have the folders 
        crd/1_2/1, crd/1_2/2, crd/1_2/3
        This would return 4
    Parameters: struPath, the path that we are looking at. So for the example in the 
        description, this would be 'crd/1_2'
    Returns: next_frame, which is the next non-existent folder (4 in the example)
    '''
    next_frame = 1
    while True:
        pdbPath = struPath + '/' + str(next_frame) 
        if os.path.exists(pdbPath):
            next_frame += 1
        else:
            return next_frame - 1
            
def milestoning(parameter, skip_compute):
#    import multiprocessing as mp
    import pandas as pd
    ms = milestone()
    data_path = parameter.crdPath
    outputpath = parameter.currentPath
    create_folder(outputpath)
    ms.read_anchors(parameter.AnchorPath)
    
    files = ['/k.txt', '/k_norm.txt', '/life_time.txt', '/info.txt', '/results.txt', '/list.txt', '/ms_index.npy', '/log', '/committor.txt', '/enhanced_count', '/t.txt']
#    backup(files)
    
    path_list = []
    anchor_orig_list = []
    enhanced_count = {}
    total_trajs = {}
    for anchor1 in range(1, parameter.AnchorNum + 1):
        for anchor2 in range(anchor1 + 1, parameter.AnchorNum + 1):
            MSname = str(anchor1) + '_' + str(anchor2)
            MSpath = parameter.crdPath + "/" + MSname
            if not os.path.exists(MSpath):
                continue
            #if 'MS' + MSname in parameter.skip_MS:
            #    continue

            for config in range(1, parameter.nframe):
                current_filepath = MSpath + '/' + str(parameter.iteration) + '/' + str(config)
                if parameter.software == 'namd':
                    traj_filepath = current_filepath + '/stop.colvars.state'
                else:
                    traj_filepath = current_filepath + "/" + parameter.outputname + ".colvars.traj" 

                if not os.path.isfile(traj_filepath) or os.stat(traj_filepath).st_size == 0:
                    continue
                if os.path.isfile(current_filepath + '/enhanced'):
                    try:
                        enhanced_count[MSname] += 1
                    except:
                        enhanced_count[MSname] = 1
                path_list.append(traj_filepath)
                if os.path.exists(current_filepath + '/end.txt'):
                    try:
                        total_trajs[MSname] += 1
                    except:
                        total_trajs[MSname] = 1
#                anchor_orig_list.append([anchor1, anchor2])
#    print(total_trajs)
    for path in path_list:
        
        start_info = os.path.dirname(os.path.abspath(path)) + '/start.txt'
        end_info = os.path.dirname(os.path.abspath(path)) + '/end.txt'
        time_info = os.path.dirname(os.path.abspath(path)) + '/lifetime.txt'
        
#        print(path)

        #milestones(parameter).get_final_ms(os.path.dirname(os.path.abspath(path)))
        
        if not os.path.isfile(start_info) or os.stat(start_info).st_size == 0:
            continue
        if not os.path.isfile(end_info) or os.stat(end_info).st_size == 0:
            continue
        if not os.path.isfile(time_info) or os.stat(time_info).st_size == 0:
            continue
        
        start = pd.read_csv(start_info, header=None, delimiter=r'\s+').values.tolist()[0]
        end = pd.read_csv(end_info, header=None, delimiter=r'\s+').values.tolist()[0]
        time = pd.read_csv(time_info, header=None, delimiter=r'\s+').values.tolist()[0]
#        print(time)
        #print(time)
       # if parameter.software == 'gromacs':
       #     time[0] *= 1000
        if start == end or end == [0,0] or 0 in end:
            continue
        if float(time[0]) == 0:
            continue
        elif parameter.max_lifetime is not None:
            if float(time[0]) > parameter.max_lifetime:
                continue

#       ignore new milestones
        name_orig = 'MS' + str(start[0]) + '_' + str(start[1])
        if parameter.ignorNewMS and name_orig not in parameter.MS_list:
            continue
        name_dest = 'MS' + str(end[0]) + '_' + str(end[1])
        if parameter.ignorNewMS and name_dest not in parameter.MS_list:
            continue
            
        if parameter.ignore_transitions:
            skip_traj = False
            for transitions in parameter.ignore_transitions:
#                print(start, transitions[0], end, transitions[1])
                if name_orig == 'MS' + transitions[0] and name_dest == 'MS' + transitions[1]:
                    skip_traj = True
            if skip_traj == True:
                continue

        if str(start[0]) + '_' + str(start[1]) not in parameter.network.keys():
            parameter.network[str(start[0]) + '_' + str(start[1])] = set()
        parameter.network[str(start[0]) + '_' + str(start[1])].add(str(end[0]) + '_' + str(end[1]))
        
        anchor_orig = []
        anchor_dest = []
        lifetime = []
        
        anchor_orig.append([int(start[0]), int(start[1])])
        anchor_dest.append([int(end[0]), int(end[1])])
        lifetime.append(float(time[0]))
#        time.append(int(result[i, 5]))
         
#        del_restarts(parameter, os.path.dirname(os.path.abspath(path)), time, 100000, 100)
        
        if len(anchor_dest) == 0:
            continue
        
        for i in range(len(anchor_orig)):    
            name_orig = str(anchor_orig[i][0]) + '_' + str(anchor_orig[i][1])
            name_dest = str(anchor_dest[i][0]) + '_' + str(anchor_dest[i][1])
            
            ms.add_ms(anchor_orig[i][0], anchor_orig[i][1], 'orig')
            ms.add_ms(anchor_dest[i][0], anchor_dest[i][1], 'dest')
            
            index1 = [k for k,v in ms.ms_index.items() if v == anchor_orig[i]]
            index2 = [k for k,v in ms.ms_index.items() if v == anchor_dest[i]]
            '''
            total_frames = 0
            temp_path = parameter.crdPath + '/' + str(name_orig) + '/' + str(parameter.iteration)
            total_trajs = get_traj_amount(temp_path)
            for j in range(1, total_trajs+1):
                if os.path.exists(temp_path + '/' + str(j) + '/stop.colvars.state'):
                    total_frames += 1
            '''
            frame_value = total_trajs[name_orig]
            ms.k_count[index1, index2] += 1
            '''
            if parameter.software == 'namd':
                lifetime_val = lifetime[i]
            else:
                lifetime_val = lifetime
            '''
            ms.t_count[index1, index2] += (lifetime[0]/frame_value)

            if ms.t_hash.get(name_orig):
                new_slice = str(ms.t_hash.get(name_orig)) + "," + str(lifetime[i])
            else:
                new_slice = str(lifetime[i])
            ms.t_hash[name_orig] = new_slice
            #print(new_slice)  

    t = np.zeros((len(ms.ms_index), 1))
    t_std = np.zeros((len(ms.ms_index), 1))  
    
    for i in range(len(t)):
        name = str(ms.ms_index.get(i)[0]) + '_' + str(ms.ms_index.get(i)[1])
        if ms.t_hash.get(name):
            time_list = list(map(float, ms.t_hash.get(name).split(',')))
            #print(time_list)
            t[i, 0] = np.mean(time_list)
            if len(time_list) > 0:
                t_std[i, 0] = (np.std(time_list, ddof=1))/sqrt(len(time_list))
            else:
                t_std[i, 0] = 0
                log("Only one life_time value for {}, standard deviation not calculated".format(name)) 
    
    if skip_compute == False:
        for i in (0,1):
            below_min = []
            k_sums = np.sum(ms.k_count,axis=i)
            for i in range(len(k_sums)):
                if k_sums[i] <= parameter.k_min_sum:
                    below_min.append(i)
            to_delete = below_min
            to_delete.reverse()
            if to_delete:
                print(to_delete)
                reorder = True
                ms.k_count = np.delete(ms.k_count, to_delete, axis=1)
                ms.k_count = np.delete(ms.k_count, to_delete, axis=0)    
                for k in to_delete:
                    ms.ms_index.pop(k)
                    for j in range(k,len(ms.ms_index.keys())):
                        ms.ms_index[j] = ms.ms_index.pop(j+1)  
                #ms.ms_index = np.array(np.delete(ms.ms_index, idx, axis=1))        
                t = np.delete(t, to_delete, axis=0)        
                t_std = np.delete(t_std, to_delete, axis=0)         
                #ms.ms_index = np.delete(ms.ms_index, idx, axis=1)
                    
    k_ordered, t_ordered, t_std_ordered, index_ordered, t_matrix_ordered = K_order(ms.k_count, t, t_std, ms.ms_index, ms.t_count, parameter)
    ms.k_count = k_ordered.copy()

    t = t_ordered.copy()
    t_std = t_std_ordered.copy()
    ms.ms_index = index_ordered.copy()
    ms.t_count = t_matrix_ordered.copy()    
    
    with open(outputpath + '/k.txt', 'w+') as fk:
        m = ['{}_{}'.format(item[0], item[1]) for item in list(ms.ms_index.values())]
        print(''.join(['{:>10}'.format(item) for item in m]),file=fk)
        print('',file=fk)
        print('\n'.join([''.join(['{:10d}'.format(item) for item in row])for row in ms.k_count]),file=fk)

    with open(outputpath + '/t.txt', 'w+') as ft:
        m = ['{}_{}'.format(item[0], item[1]) for item in list(ms.ms_index.values())]
        print(''.join(['{:>10}'.format(item) for item in m]),file=ft)
        print('',file=ft)
        print('\n'.join([''.join(['{:10.2f}'.format(item) for item in row])for row in ms.t_count]),file=ft)
        
    with open(outputpath + '/life_time.txt', 'w+') as f1:
        m = ['{}_{}'.format(item[0], item[1]) for item in list(ms.ms_index.values())]
        print(''.join(['{:>10}'.format(item) for item in m]),file=f1)
        print('',file=f1)
        if parameter.software == 'namd':
            print('\n'.join([''.join(['{:10.2f}'.format(item) for item in np.squeeze(t)])]),file=f1)
            print('\n'.join([''.join(['{:10.2f}'.format(item) for item in np.squeeze(t_std)])]),file=f1)
        else:
            print('\n'.join([''.join(['{:10.2f}'.format(item) for item in np.squeeze(t)])]),file=f1)
            print('\n'.join([''.join(['{:10.2f}'.format(item) for item in np.squeeze(t_std)])]),file=f1)
  
    k_ave = k_average(ms.k_count)    
    with open(outputpath + '/k_norm.txt', 'w+') as f1:
        f1.write('\n'.join([''.join(['{:10.5f}'.format(item) for item in row])for row in k_ave]))   
    np.save(outputpath + '/ms_index.npy', ms.ms_index)
    
    if skip_compute == False:
        compute(parameter, False)
        log("Computing finished. Mean first passage time: {:20.7f} ps".format(parameter.MFPT))
    else:
        log("Computing skipped because new milestones were found")
    backup(parameter, files)
    return ms, ms.new, ms.known


if __name__ == '__main__':
    from parameters import *
    from milestones import *
    new = parameters()
    new.initialize()
    new.MS_list = milestones(new).read_milestone_folder()
    print(new.MS_list)
    new.iteration = 1
    milestoning(new, False)

