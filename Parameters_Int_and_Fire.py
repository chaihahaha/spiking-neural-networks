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

import datetime
now = datetime.datetime.now() # gives me date and time

file=open("Parameters.txt","w")
file.write("Parameters used in this simulation:\n")
file.write(" \n") # empty line
file.write("Date: " + str(now) + "\n")
file.write(" \n") # empty line
file.write("tau_mem= %(tau_mem)f\n" % {"tau_mem":tau_mem})
file.write("E_leak= %(E_leak)f\n" % {"E_leak":E_leak})
file.write("V_reset= %(V_reset)f\n" % {"V_reset":V_reset})
file.write("V_thresh= %(V_thresh)f\n" % {"V_thresh":V_thresh})
file.write("E_e= %(E_e)f\n" % {"E_e":E_e})
file.write("tau_e= %(tau_e)f\n" % {"tau_e":tau_e})
file.write("firing_rate_e= %(firing_rate_e)f\n" % {"firing_rate_e":firing_rate_e})
file.write("w_e= %(w_e)f\n" % {"w_e":w_e})
file.write("numb_exc_syn= %(numb_exc_syn)f\n" % {"numb_exc_syn":numb_exc_syn})
file.write("E_i= %(E_i)f\n" % {"E_i":E_i})
file.write("tau_i= %(tau_i)f\n" % {"tau_i":tau_i})
file.write("firing_rate_i= %(firing_rate_i)f\n" % {"firing_rate_i":firing_rate_i})
file.write("w_i= %(w_i)f\n" % {"w_i":w_i})
file.write("numb_inh_syn= %(numb_inh_syn)f\n" % {"numb_inh_syn":numb_inh_syn})
file.write("t_0= %(t_0)f\n" % {"t_0":t_0})
file.write("t_max= %(t_max)f\n" % {"t_max":t_max})
file.write("time_step_sim= %(time_step_sim)f\n" % {"time_step_sim":time_step_sim})
file.write("delta_t= %(delta_t)f\n" % {"delta_t":delta_t})
file.write(" \n") # empty line
file.write("STDP Parameters:\n")
file.write("tau_LTP= %(tau_LTP)f\n" % {"tau_LTP":tau_LTP})
file.write("A_LTP= %(A_LTP)f\n" % {"A_LTP":A_LTP})
file.write("tau_LTD= %(tau_LTD)f\n" % {"tau_LTD":tau_LTD})
file.write("A_LTD= %(A_LTD)f\n" % {"A_LTD":A_LTD})
file.write("w_max= %(w_max)f\n" % {"w_max":w_max})
file.write("c1= %(c1)f\n" % {"c1":c1})
file.write("c2= %(c2)f\n" % {"c2":c2})
file.write("tau_c= %(tau_c)f\n" % {"tau_c":tau_c})
file.close()