# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 09:52:52 2017
Tested in Python 3.6.5, Anaconda Inc.

@author: Miehl
Adapted by: Florence Kleberg

"""

class Correlated_Trains:
    
    """
    Class for creating groups of spike trains, with
    within-group correlations created by copying spikes.
    These correlations are instantaneous (no jitter) and Poisson
    statistics are maintained at the single train level.
    Based on Romain Brette, 2008. 
    """
    
    def get_list_of_trains(self,c1,c2,frate):
        import Parameters_Int_and_Fire
        import numpy as np
        import numpy.random as nr
        
        nr.seed() # seeds with time
        
        # get parameters
        time_step_sim=Parameters_Int_and_Fire.time_step_sim
        t_max=Parameters_Int_and_Fire.t_max
        t_0=Parameters_Int_and_Fire.t_0
        nin_tot = Parameters_Int_and_Fire.numb_exc_syn
        
        tt1=t_0
        
        firing_rate_target=frate
        
        # create two groups of correlated Poisson spike trains.
        for group in range(2):
            
            spike_time_source=[]
            list_of_all_spike_trains=[]
            tt1=t_0
            
            # calculate probability to copy a spike-time
            if group==0:
                copy_probab=np.sqrt(c1)
            else:
                copy_probab=np.sqrt(c2)
            
            # gnerate a source spike train (Poisson)
            rnd = nr.random(int(np.round(t_max/time_step_sim)))
            spikes = rnd < (firing_rate_target/1000) * time_step_sim
            spike_time_source = (np.where(spikes)[0] * time_step_sim).tolist()
        
            list_of_all_spike_trains.append(spike_time_source)
                    
            #generate the correlated spike trains in the group
            for syn in range(1,int(nin_tot/2)):
            
                spike_time_new=[]
                tt2=t_0
               
                # copy some of the spike times
                for ii in range(1,len(spike_time_source)+1):
                    if nr.uniform(0,1)<=copy_probab:
                        spike_time_new.append(spike_time_source[ii-1])
                        
                        ### jitter spike correlations:
                        # tt2=tt1
                        #tau_c=3
                        #tt2=tt1+1/tau_c*np.exp(-random.uniform(-10,10)/tau_c)                    
                        #spike_time_source.append(tt2)
                
                #calculate the required noise firing rate
                firing_rate_noise=firing_rate_target-len(spike_time_source)/t_max*1000*time_step_sim*copy_probab
                # fill the spike train with the new 'noise' spikes
                rnd = nr.random(int(np.round(t_max/time_step_sim)))
                spikes = rnd < (firing_rate_noise/1000) * time_step_sim
                spike_time_noise = (np.where(spikes)[0] * time_step_sim).tolist()
                spike_time_all = spike_time_new + spike_time_noise
                 
                # if one spike time is per chance twice in the list, erase it
                spike_time_new2=list(set(spike_time_all))
                # sort the list of spike times
                spike_time_new3=sorted(spike_time_new2)
                
                # store the spiketrain
                list_of_all_spike_trains.append(spike_time_new3)
        
            if group==0:
                list_of_all_spike_trains1=list_of_all_spike_trains
            elif group==1:
                list_of_all_spike_trains2=list_of_all_spike_trains
            else:
                print("Error in Correlated_Spike_Trains")
                
              # two groups of spiketrains are returned.    
        return [list_of_all_spike_trains1,list_of_all_spike_trains2]
