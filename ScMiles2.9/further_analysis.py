# -*- coding: utf-8 -*-
"""
Created on Tue May  4 10:33:35 2021

@author: allis
"""

#further analysis
import os
import networkx as nx
from compute import *
from milestoning_mp import *
from parameters import *
from network_check import *
from math import sqrt
import matplotlib.pyplot as plt
import math

class analysis:
    def __init__(self, parameter, k_cutoff=None, max_lifetime=None, custom_k=None, max_flux=None, starting_iteration=None):
        self.parameter = parameter
        self.k_min_sum = 0
        self.max_lifetime = None
        self.data_file = False
        self.source = False
        self.iteration = 1
        self.reactant = None
        self.product = None
        self.max_flux = False
        self.milestone_list = []
        self.plot_network = False
        self.pbc = []
        self.all_iterations = False

    def read_analysis(self):
        if not os.path.isfile(self.parameter.inputPath + '/analysis.txt'): 
            print('No file called analysis.txt in my_project_input. Unable to do analysis')
            return
        self.parameter.initialize()

        with open(self.parameter.inputPath + '/analysis.txt') as f:
            for line in f:
                line = line.split()
                if line == []:
                    continue
                if 'source' in line[0]:
                    if line[1].lower() == 'scmiles':
                        self.source = 'scmiles'
                    elif line[1].lower() == 'path_history.dat': 
                        self.source = 'path_history'
                    elif line[1].lower() == 'custom':
                        self.source = 'custom'
                    elif '.colvars.traj' in line[1].lower():
                        self.source = 'colvar_file'
                if 'pbc' in line[0]:
                    self.pbc = line[1].split(',')
                    for i in range(len(self.pbc)):
                        self.pbc[i] = float(self.pbc[i])
                if 'k_min_sum' in line[0]:
                    self.k_min_sum = int(line[1])
                    self.parameter.k_min_sum = int(line[1])
                if 'ignore_milestones' in line:
                    self.parameter.analysis_ignore_milestones = line[1].split(',')
                if 'ignore_transitions' in line: 
                    transitions = line[1].split(',')
                    for i in range(len(transitions)):
                       transitions[i] = transitions[i].split('-')
                    self.parameter.ignore_transitions = transitions 
                if 'max_lifetime' in line[0]:
                    self.parameter.max_lifetime = float(line[1])
                if 'all_iterations' in line[0]:
                    self.parameter.running_average = True
                    self.all_iterations = True                    
                if 'data_file' in line[0]:
                    self.data_file=True
                if  line[0] == 'iteration':
                    self.iteration = int(line[1])
                    self.parameter.iteration = int(line[1])
                if 'reactant' in line[0]:
                    r = line[1].split(',')
                    for i in range(len(r)):
                        r[i] = int(r[i])
                    self.parameter.reactant = r
                if 'product' in line[0]:
                    r = line[1].split(',')
                    for i in range(len(r)):
                        r[i] = int(r[i])
                    self.parameter.product = r  
                if 'max_flux' in line:
                    self.max_flux = True
                if 'MS_list' in line:
                    ms_index = dict()
                    ms = line[1].split(',')
                    for i in range(len(ms)):
                        self.parameter.MS_list.add('MS' + ms[i])
                        self.milestone_list.append(ms[i])
                        ms_index[i] = get_anchors(ms[i])
                if 'plot_network' in line:
                    self.plot_network = True
        if not self.all_iterations:
            self.parameter.starting_iteration = self.parameter.iteration
        else:
            self.parameter.starting_iteration = 1
        if self.source == 'path_history':
            self.long_traj()
        elif self.source == 'scmiles':
            self.parameter.MS_list = milestones(self.parameter).initialize(status=1)
            if self.parameter.analysis_ignore_milestones:
                remove_ms = []
                for i in self.parameter.MS_list:
                    if i in self.parameter.analysis_ignore_milestones:
                        remove_ms.append(i)
                    self.milestone_list.append(i)
                if remove_ms != []:
                    for i in remove_ms:
                        self.parameter.MS_list.remove(i)
            #network_check(self.parameter.MS_list)
            milestoning(self.parameter, False)
        elif self.source == 'colvar_file':
            self.colvar_file()
        elif self.source == 'custom':
            np.save(self.parameter.currentPath + '/ms_index.npy', ms_index)
            compute(parameter)
        if self.plot_network == True:
            self.plot()
        if self.data_file == True:
            self.make_data_file()

            
        #if self.max_flux == True:
        #    self.max_flux_path()
                   
    def find_last_iteration(self, path):
        iteration = 1
        while True:
            if os.path.exists(path + '/' + str(iteration)):
                iteration += 1
            else:
                return iteration
            
    def make_data_file(self):
        import pandas as pd
        import os
        import re
        data = []
        for ms in self.parameter.MS_list:
            lst = re.findall('\d+',ms)
            name = lst[0] + '_' + lst[1]
            ms_path = self.parameter.crdPath + '/' + name
            iteration = self.find_last_iteration(ms_path)
            next_frame = get_next_frame_num(ms_path)
            for i in range(1, iteration):
                for traj in range(1, next_frame):
                    tmp = []
                    traj_path = ms_path + str(i) + '/' + str(traj) 
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
                    data.append([parameter.iteration, start[0], start[1], end[0], end[1], lifetime[0], enhanced, traj_path])
        with open(parameter.currentPath + '/all_data.txt', 'w+') as f1:
            for item in data:
                f1.write(" ".join(map(str,item)) + '\n')
           
    def colvar_file(self):
        raw_data = []
        times = [0,0]
        anchors = pd.read_csv(self.parameter.inputPath + "/anchors.txt", header=None, delimiter=r"\s+").values.tolist()       
        anchor_number = len(anchors)
        current_anchors = [0,0]
        transitions = []
        milestones = set()
        first = True
        with open(self.parameter.inputPath + "/analysis.colvars.traj") as f:
            for line in f:
                if line.startswith('#'):
                    continue
                line=line.split()
                for i in range(len(line)):
                    line[i] = float(line[i])
                raw_data.append(line)
                if self.parameter.software == 'gromacs':
                    line[0] = line[0]*1000
                all_rmsd = [0] * anchor_number
                for an in range(anchor_number):
                    rmsd = 0
                    for colvar in range(1,len(line)):
                        if self.pbc:
                            if self.pbc[colvar-1] != 0:
                                value = abs(line[colvar] - anchors[an][colvar-1])
                                if value > self.pbc[colvar-1]/2:
                                    value = self.pbc[colvar-1] - value
                        else:
                            value = line[colvar] - anchors[an][colvar-1]
                        rmsd += (value)**2
                    rmsd = sqrt(rmsd)
                    all_rmsd[an] = rmsd
                anchor = all_rmsd.index(min(all_rmsd)) + 1
                times[1] = line[0]
                if first == True:
                    times[0] = line[0]
                    all_rmsd[anchor-1] = max(all_rmsd) #just to find the second largest
                    first = False
                    current_anchors[1] = anchor
                    continue
                if anchor in current_anchors:
                    if anchor == current_anchors[0]:
                         current_anchors.reverse()
                    continue
                anchors_sorted = sorted([anchor, current_anchors[1]])
                #if self.parameter.analysis_ignore_milestones:
                #    if 'MS' + str(anchors_sorted[0]) + '_' + str(anchors_sorted[1]) in self.parameter.analysis_ignore_milestones:
                #        continue
                previous_ms = sorted(current_anchors)
                new_ms = anchors_sorted
                current_anchors = [current_anchors[1],anchor]
                if 0 in previous_ms:
                    continue
                milestones.add(str(new_ms[0]) + '_' + str(new_ms[1]))
                transitions.append([str(previous_ms[0]) + '_' + str(previous_ms[1]), str(new_ms[0]) + '_' + str(new_ms[1]),round(times[1]-times[0],5)])
                times[0] = times[1]
        self.create_files(milestones, transitions)
    
        
    def long_traj(self):
        count = 0
        create_folder(self.parameter.outputPath)
        create_folder(self.parameter.currentPath)
        raw_data = pd.read_csv(self.parameter.inputPath + "/path_history.dat", header=None, delimiter=r"\s+").values.tolist()            
        milestones = set()
        anchors = [0,0]
        times = [0,0]
        transitions = []
        previous_milestone = str(int(raw_data[0][1]) + 1) + '_' + str(int(raw_data[1][1] + 1))
        times[0] = (raw_data[1][0] * 1000000)
        for i in range(len(raw_data)):
            anchors[0] = anchors[1]
            anchors[1] = int(raw_data[i][1] + 1)
            times[1] = (raw_data[i][0] * 1000000)
            ms_int_form = sorted(anchors)
            ms = str(ms_int_form[0]) + '_' + str(ms_int_form[1])
            #if self.parameter.analysis_ignore_milestones:
            #    if 'MS' + ms in self.parameter.analysis_ignore_milestones:
            #        continue
            if 0 in ms_int_form:
                continue
            milestones.add(ms)
            if ms != previous_milestone:
                transitions.append([previous_milestone, ms, round(times[1]-times[0],5)])
                times[0] = times[1]
                previous_milestone = ms
        print(milestones)
        self.create_files(milestones, transitions)
        
    def create_files(self, milestone_set, transitions):
        milestones = []
        for i in milestone_set:
            ms = get_anchors(i)
            milestones.append(ms)
        lifetimes = dict()
        ms_index = dict()       
        milestones = sorted(milestones)
        for i in range(len(milestones)):
            milestones[i] = str(milestones[i][0]) + '_' + str(milestones[i][1])
            lifetimes[milestones[i]] = []
        matrix = np.zeros((len(milestones),len(milestones)))
        t_matrix = np.zeros((len(milestones),len(milestones)))
        for i in range(len(transitions)):
            if parameter.analysis_ignore_milestones:
                if 'MS' + transitions[i][0] in self.parameter.analysis_ignore_milestones or 'MS' + transitions[i][1] in parameter.analysis_ignore_milestones:
                    continue
            matrix[milestones.index(transitions[i][0]),milestones.index(transitions[i][1])] += 1
            t_matrix[milestones.index(transitions[i][0]),milestones.index(transitions[i][1])] += transitions[i][2]
            lifetimes[transitions[i][0]].append(transitions[i][2])
        printing_lifetimes = [[],[],[]]
        for i in range(len(milestones)):
            for j in range(len(milestones)):
                t_matrix[i][j] = float(t_matrix[i][j]/len(lifetimes[milestones[i]]))
        for i in range(len(milestones)):
            printing_lifetimes[0].append(milestones[i])
            printing_lifetimes[1].append(np.mean(lifetimes[milestones[i]]))
            if len(lifetimes[milestones[i]]) > 1:
                printing_lifetimes[2].append((np.std(lifetimes[milestones[i]], ddof=1))/sqrt(len(lifetimes[milestones[i]])))
            else:
                printing_lifetimes[2].append(0)
            
        '''
        sum_mat = 0
        for i in range(len(matrix[0])):
            for j in range(len(matrix)):
                sum_mat += matrix[i][j]
        '''
        if self.k_min_sum is not None:
            while True:
                milestones_to_delete = []
                remove_index = []
                for i in range(len(milestones)):
                    total = 0
                    for j in range(len(milestones)):
                        total += matrix[j][i]
                    if total <= self.k_min_sum:
                        remove_index.append(i)
                        milestones_to_delete.append(milestones[i])
                if remove_index:
                    remove_index.reverse()
                    for i in remove_index:
                        for j in range(len(printing_lifetimes)):
                            printing_lifetimes[j].pop(i)
                        matrix = np.delete(matrix,i,0)
                        matrix = np.delete(matrix,i,1)                
                        t_matrix = np.delete(matrix,i,0)
                        t_matrix = np.delete(matrix,i,1)
                        milestones.pop(i)
                else:
                    break
                    
        with open(self.parameter.currentPath + '/life_time.txt', 'w+') as f:  
            print(''.join(['{:>10}'.format(item) for item in printing_lifetimes[0]]),file=f)
            print('',file=f)
            print('\n'.join([''.join(['{:10.2f}'.format(item) for item in np.squeeze(printing_lifetimes[1])])]),file=f)
            print('\n'.join([''.join(['{:10.2f}'.format(item) for item in np.squeeze(printing_lifetimes[2])])]),file=f)
            
        with open(self.parameter.currentPath + '/k.txt', 'w+') as f:
            print(''.join(['{:>10}'.format(item) for item in printing_lifetimes[0]]),file=f)
            print('',file=f)
            print('\n'.join([''.join(['{:10d}'.format(int(item)) for item in row])for row in matrix]),file=f)
        
        with open(self.parameter.currentPath + '/t.txt', 'w+') as f:
            print(''.join(['{:>10}'.format(item) for item in printing_lifetimes[0]]),file=f)
            print('',file=f)
            print('\n'.join([''.join(['{:10.2f}'.format(int(item)) for item in row])for row in t_matrix]),file=f)

        k_ave = k_average(matrix)
        with open(self.parameter.currentPath + '/k_norm.txt','w+') as f:
            f.write('\n'.join([''.join(['{:10.5f}'.format(item) for item in row])for row in k_ave]))
        
        for i in range(len(milestones)):
            [anchor1, anchor2] = get_anchors(milestones[i])
            ms_index[i] = sorted([anchor1,anchor2])
        np.save(self.parameter.currentPath + '/ms_index.npy', ms_index)
        self.milestone_list = milestones
        k, index, q = compute(parameter)

         
    def plot(self):
        #number_of_anchors=43
        radius_of_circle=10
        #milestones=[1.02, 1.03, 1.04, 1.05, 1.08, 2.04, 2.05, 2.07, 1.06, 3.06, 3.08, 3.1, 3.04, 4.05, 4.06, 4.07, 3.05, 5.07, 5.08, 5.09, 1.07, 1.12, 10.12, 6.07, 6.1, 6.11, 6.12, 8.09, 8.1, 12.13, 3.07, 3.09, 3.11, 9.11, 10.11, 9.1, 11.12, 11.13, 13.14, 11.14, 13.16, 14.15, 14.16, 15.16, 16.17, 17.18, 16.18, 16.22, 16.29, 22.29, 25.29, 27.29, 18.19, 18.21, 17.21, 21.22, 17.27, 21.27, 27.31, 18.2, 19.2, 20.21, 21.26, 18.27, 26.27, 19.24, 20.23, 19.21, 19.26, 19.28, 23.24, 24.25, 24.26, 24.35, 26.28, 28.31, 28.35, 20.22, 22.23, 20.26, 22.26, 23.25, 22.25, 25.26, 25.3, 23.26, 26.29, 29.3, 24.28, 25.28, 28.3, 31.35, 35.37, 35.39, 26.3, 30.31, 27.28, 27.3, 29.31, 31.32, 31.33, 33.35, 34.35, 35.36, 35.41, 32.33, 32.34, 32.42, 33.34, 34.38, 32.35, 41.42, 34.36, 34.37, 36.37, 36.38, 37.38, 37.39, 34.4, 38.39, 38.4, 38.41, 39.4, 40.41, 40.42, 42.43]
        number_of_anchors = 0
        milestones = []
        for i in self.milestone_list:
            anchors = get_anchors(i)
            number_of_anchors = max(anchors[0],anchors[1],number_of_anchors)
            if anchors[1] < 10:
                anchors[1] = '0' + str(anchors[1])
            else:
                anchors[1] = str(anchors[1])
            ms = float(str(anchors[0]) + '.' + anchors[1])
            milestones.append(ms)
        
        fig = plt.figure(figsize=(10,10),dpi=200)
        ax = fig.add_subplot(111)#, projection='3d')
        ax.set_ylim([-12,12])
        ax.set_xlim([-12,12])
        ax.tick_params(axis='x', labelsize=0)
        ax.tick_params(axis='y', labelsize=0)
        ax.set_xlabel('x',fontsize=10)
        ax.set_ylabel('y',fontsize=10)
        
        ### code to generate points on a circle centered at 0,0
        pi=math.pi
        def PointsInCircum(r,n=number_of_anchors):
            return [(math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r) for x in range(0,n+1)]
        point=PointsInCircum(radius_of_circle)
        for i in range(len(point)-1):
            if point[i][0] >0:
                x = 50
            else:
                x = -50
            if point[i][1] > 0:
                y = 50
            else:
                y = -50
            ax.annotate(i+1, (point[i][0], point[i][1]), xytext=(x, y), textcoords='offset pixels')
        
        
        ### plot the circle;  this isn't  required. Tried just check the correctness. 
        circle1=plt.Circle((0, 0), 10, color='none',fill=False,linewidth=4.0)
        
        
        #### code to plot the connections between the anchors
        for i in milestones:
            #### extracting anchor numbers from the milestone name so from milestone 11.20 getting anchor 11 and anchor 20
            a=int((i*100)//100)
            b=int(round((i*100)%100))
            x_1=point[a-1][0]
            y_1=point[a-1][1]
            x_2=point[b-1][0]
            y_2=point[b-1][1]
            l_x=[x_1,x_2]
            l_y=[y_1,y_2]
            ax.plot(l_x,l_y,c='k',marker='o',markersize=3,linewidth=2.0)
        
        ax.add_artist(circle1)
        
        ##### plot the points on the circle
        for i in range(len(point)):
            ax.scatter(point[i][0], point[i][1], c = 'k', marker='o',s=100)
        '''     
        ax.scatter(point[0][0], point[0][1], c = 'r', marker='o',s=300)
        ax.scatter(point[number_of_anchors-1][0], point[number_of_anchors-1][1], c = 'g', marker='o',s=300)
        '''
        for i in self.parameter.reactant:
            ax.scatter(point[i-1][0], point[i-1][1], c = 'g', marker='o',s=300)
        for i in self.parameter.product:
            ax.scatter(point[i-1][0], point[i-1][1], c = 'r', marker='o',s=300)
        create_folder(parameter.currentPath + '/plots')
        plt.savefig(parameter.currentPath + '/plots/' + 'network_analysis' + '.png')



#plt.grid()
plt.show()


if __name__ == '__main__':
    from milestoning_mp import *
    from milestones import *

    parameter = parameters()
    analyze = analysis(parameter)
    analyze.read_analysis()
    
    
