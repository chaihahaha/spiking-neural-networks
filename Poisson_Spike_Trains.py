# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 09:52:52 2017
Tested in Python 3.6.5, Anaconda Inc.

@author: Miehl
Adapted by: Florence Kleberg
"""

import numpy as np 

class Poisson_Trains:
    
    """
    This class creates separate groups of stationary Poisson spike trains.
    Each group has its own firing rate.
    """
    
    @classmethod # this method is accessed by other methods within this class.
    def makePoisson(self,nu,dur,res):
        rnd = np.random.random(dur * res)
        spikes = rnd < nu / res
        times = np.where(spikes)[0] / res
        return times
    
    def get_list_of_trains(self, r1, r2):
        import Parameters_Int_and_Fire
        import random
        import numpy
        
        random.seed() # seeds with time
        
        # get parameters
        ngroups = 2
        firing_rate_e = [r1, r2]
        time_step_sim = Parameters_Int_and_Fire.time_step_sim
        t_max = Parameters_Int_and_Fire.t_max
        #t_0 = Parameters_Int_and_Fire.t_0
        nin_tot = Parameters_Int_and_Fire.numb_exc_syn
                
        # create two groups of independent Poisson spike trains.
        for group in range(ngroups):
            
            list_of_all_spike_trains=[]
                            
            #generate all spike trains for this group
            for syn in range(1,int(nin_tot/2 + 1)):
            
                spike_times_new = Poisson_Trains.makePoisson((firing_rate_e[group]*time_step_sim/1000),t_max,time_step_sim)
                
                # if one spike time is per change twice in the list, erase it
                spike_times_new2=list(set(spike_times_new))
                # sort the list of spike times
                spike_times_new3=sorted(spike_times_new2)
                
                # store the spiketrain
                list_of_all_spike_trains.append(spike_times_new3)
        
            if group==0:
                list_of_all_spike_trains1=list_of_all_spike_trains
            elif group==1:
                list_of_all_spike_trains2=list_of_all_spike_trains
            else:
                print("Error in Poisson_Spike_Trains. Cannot have >2 groups.")
                
              # two groups of spiketrains are returned.    
        return [list_of_all_spike_trains1,list_of_all_spike_trains2]
