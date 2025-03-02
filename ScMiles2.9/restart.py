# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:41:42 2020

@author: allis
"""
import glob
from log import *
from milestones import *

def read_log(parameter, status):
    import os
    current_iteration = []
    lastLog = ""
    if parameter.milestone_search == 1 or parameter.milestone_search == 3:
        seek = False
    else:
        seek = True
    if status == 1:
        seek = True
        sample = True
    else:
        sample = False
    #uses log to find latest iteration
    with open(file = os.path.join(parameter.currentPath, 'log')) as r:
        for line in r:
            if line == "":
                continue
            line = line.rstrip()
            info = line.split()
            info = info[2:]
            if 'Iteration' in line and 'complete' not in line and 'created' not in line:
                current_iteration = []
                #current_iteration.append(" ".join(str(x) for x in info))
                parameter.iteration = int(info[2])
                seek = True
                sample = True
            elif 'Reactant and product are connected' in line:
                #We are done seeking
                seek = True
            elif 'Initial sampling completed' in line:
                #We are done sampling
                sample = True
                seek = True
            current_iteration.append(" ".join(str(x) for x in info))
            lastLog = (" ".join(str(x) for x in info))
    if parameter.iteration != 0:
        parameter.iteration -= 1
    log("ScMiles restarted")
#    print(parameter.iteration)
#    find_log_code(lastLog)
    #print(lastLog)
    if parameter.method == 1:
        for i in current_iteration:
            if 'skip ms' in i.lower():
                info = i.split()
                parameter.skip_MS.append(info[1])
    return seek, sample, lastLog

            
            

if __name__ == '__main__':
    from parameters import *
    from namd_conf_custom import *
    new = parameters()
    new.initialize()
    restart_simulation(new, 0)
    print(new.iteration)
    
      
