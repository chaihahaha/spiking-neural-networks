# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 12:25:24 2017
Tested in Python 3.6.5, Anaconda Inc.

@author: Miehl
Adapted by: Florence Kleberg
"""

###############
# Parameter File
###############

# simulation time values for the leaky integrate-and-fire neuron:
t_0=0 # initial time
t_max=30000 # total simulationtime in ms
time_step_sim=1 # time step for the simulation in ms

# values for the leaky integrate-and-fire neuron:
tau_mem=20 # membrane time constant in ms
E_leak=-60 # reversal potential for the leak in mV
V_reset=-70 # reset value of the neuron in mV
V_thresh=-50 # threshold of the neruon in mV

# values for the excitatory conductance g_e differential equation:
E_e=0 # reversal potential for excitatory (depolarizing) inputs in mV
tau_e=3 # postsynaptic potential (PSP), in ms
firing_rate_e=3#10 # firing rate of the excitatory synapse in Hz (per second!!!!)
w_e=0.5 # strength of all the excitatory weights
numb_exc_syn=20 # number of excitatory synapses

# values for the inhibitory conductance g_i differential equation:
E_i=-80 # reversal potential for inhibitory inputs in mV
tau_i=5 # postsynaptic potential (PSP), in ms
firing_rate_i=10 # firing rate of the inhibitory synapse in Hz (per second!!!!)
w_i=1 # strength of all the inhibitory weights
numb_inh_syn=10 # number of inhibitory synapses

# values for the Euler integration
delta_t=0.01 # integration step, in ms

# values for the STDP-rule
tau_LTP= 17 #ms
A_LTP=0.02 #mV
tau_LTD=34 #ms
A_LTD= -A_LTP*0.5#0.5 #mV
w_max=40 #mV

# values for the correlated spike trains
c1 = 0.2
c2 = 0.1
tau_c = 20 # ms
