#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:52:14 2019

@author: Wei Wei

This subroutine initializes the process of analysing.

"""
from log import log
from parameters import *
from milestoning_mp import *


def analysis_kernel(parameter):
    #This is just where we do the process of analyzing our trajectories
    log("Computing...")    
    parameter.network = {}
    info, new, known = milestoning(parameter, parameter.skip_compute) #This is the milestoning_mp.py file, compute is called within this function
    #print(info)
    if len(new) != 0:
        parameter.MS_new = (parameter.MS_new | new) - known
    if parameter.sing:
        parameter.Finished = set()
    #parameter.milestone_search = 0
        
if __name__ == '__main__':
    new = parameters()
    new.initialize()
    new.skip_compute = False
#    new.iteration = 1
    from milestones import milestones
    new.k_cutoff = 1
    new.MS_list = milestones(new).read_milestone_folder()
    analysis_kernel(new)
