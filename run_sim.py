import numpy as np
import Parameters_Int_and_Fire
from Poisson_Spike_Trains import Poisson_Trains
from Correlated_Spike_Trains import Correlated_Trains
import matplotlib.pyplot as plt

tau_mem       = Parameters_Int_and_Fire.tau_mem
E_leak        = Parameters_Int_and_Fire.E_leak
E_e           = Parameters_Int_and_Fire.E_e
E_i           = Parameters_Int_and_Fire.E_i
V_reset       = Parameters_Int_and_Fire.V_reset
V_thresh      = Parameters_Int_and_Fire.V_thresh
t_0           = Parameters_Int_and_Fire.t_0
t_max         = Parameters_Int_and_Fire.t_max
time_step_sim = Parameters_Int_and_Fire.time_step_sim
numb_exc_syn  = Parameters_Int_and_Fire.numb_exc_syn
numb_inh_syn  = Parameters_Int_and_Fire.numb_inh_syn
tau_e         = Parameters_Int_and_Fire.tau_e
tau_i         = Parameters_Int_and_Fire.tau_i
firing_rate_e = Parameters_Int_and_Fire.firing_rate_e
firing_rate_i = Parameters_Int_and_Fire.firing_rate_i
w_e           = Parameters_Int_and_Fire.w_e
w_i           = Parameters_Int_and_Fire.w_i
delta_t       = Parameters_Int_and_Fire.delta_t
# STDP parameters : 
tau_LTP       = Parameters_Int_and_Fire.tau_LTP
A_LTP         = Parameters_Int_and_Fire.A_LTP
tau_LTD       = Parameters_Int_and_Fire.tau_LTD
A_LTD         = Parameters_Int_and_Fire.A_LTD
w_max         = Parameters_Int_and_Fire.w_max
# correlation in the two groups
c1            = Parameters_Int_and_Fire.c1     
c2            = Parameters_Int_and_Fire.c2
tau_c         = Parameters_Int_and_Fire.tau_c

class Euler: 
    def euler_integration(self, f, arg_for_f, y_0, t_0, time_step_sim, delta_t):
        if delta_t>=time_step_sim:
            print("ATTENTION: The time step of the simulation is smaller than the time step of the integration!")
        n=int(round(time_step_sim/delta_t)) # calculate the number of steps
        for step in range(1,n+1):
            m=f(y_0, arg_for_f)
            y_1=y_0+delta_t*m
            t_1=t_0+delta_t
            t_0=t_1
            y_0=y_1
        return y_0

class Neuron:
    def __init__(self, init_tt, init_V_tt=-70, tau_mem=20, E_leak=-60, V_thresh=-50, V_reset=-70, int_delta_t=0.01):
        self.V_tt = init_V_tt         # membrane voltage
        self.tt = init_tt             # current time
        self.tau_mem = tau_mem        # membrane time constant
        self.E_leak = E_leak          # reversal potential for the leak
        self.V_thresh = V_thresh      # membrane voltage threshold, V_tt will be reset to V_reset after reaching this threshold
        self.V_reset = V_reset        # the reset voltage
        self.delta_t = int_delta_t    # euler integration time step

        self.synapses = []            # all the input synapses that are connected to this neuron
        self.last_spike = init_tt     # the time when this neuron last spiked

        # define function to integrate the membrane voltage equation
        # the function f(V_tt) for the simplified membrane voltage equation: d V_tt / dtt = f(V_tt)
        # whose full form is: tau_mem * d V_tt / dtt = E_leak - V_tt + g_e * (E_e - V_tt) + g_i * (E_i - V_tt)
        self.func_V = lambda V_tt, syn_input_tt: (self.E_leak - V_tt + syn_input_tt)/self.tau_mem

        self.euler = Euler()          # euler integrator
    def add_synapses(self, syns):
        # add a list of input synapses to this neuron
        for syn in syns:
            self.synapses.append(syn)
            syn.set_post_neuron(self)
    def tick(self, time_step_sim):
        # simulate the neuron for one step

        # sum all the synapse inputs
        syn_input_tt = sum([syn.g_tt * (syn.E_syn - self.V_tt) for syn in self.synapses])
        # integrate the membrane voltage equation
        V = self.euler.euler_integration(self.func_V, syn_input_tt, self.V_tt, self.tt, time_step_sim, self.delta_t)
        if V < self.V_thresh:
            self.V_tt = V
        else:
            self.V_tt = self.V_reset
            self.last_spike = self.tt
        self.tt += time_step_sim

class InputNeuron:
    def __init__(self, init_tt, spike_train):
        self.tt = init_tt
        self.spike_train = spike_train # the prescribed spike train of this input neuron

        # the last time when this neuron spiked
        # this signal will be used for the conductance calculation of its synapses
        self.last_spike = init_tt

        # the next index of the spike in the spike train which is ready for spiking
        self.next_spike_idx = 0
    def tick(self, time_step_sim):
        while self.past_spike(self.next_spike_idx, self.tt, time_step_sim) and self.next_spike_idx < len(self.spike_train)-1:
            # if sim interval went past current spike, proceed to the next spike
            self.next_spike_idx += 1
        if self.cover_spike(self.next_spike_idx, self.tt, time_step_sim):
            # if the next spike is ready for spiking (covered in current simulation time interval)
            # then record it as the last spike time
            self.last_spike = self.spike_train[self.next_spike_idx]
        self.tt += time_step_sim
    def cover_spike(self, idx, tt, time_step_sim):
        return tt <= self.spike_train[idx] < tt + time_step_sim
    def past_spike(self, idx, tt, time_step_sim):
        return self.spike_train[idx] < tt
    def before_spike(self, idx, tt, time_step_sim):
        return tt + time_step_sim <= self.spike_train[idx]

class Synapse:
    def __init__(self, init_tt, init_w_tt, E_syn, tau_syn, pre_neuron, syn_type, init_g_tt=0, w_max=40, tau_LTP=17, tau_LTD=34, A_LTP=0.02, A_LTD=-0.01, int_delta_t=0.01):
        self.tt = init_tt                      # current time
        self.g_tt = init_g_tt                  # synapse conductance
        self.w_tt = init_w_tt                  # synapse weight
        self.w_max = w_max                     # max weight for clipping
        self.E_syn = E_syn                     # potential for excitatory/inhibitory (depolarizing/polarizing) inputs
        self.tau_syn = tau_syn                 # postsynaptic potential (PSP) time constant
        self.tau_prepost = tau_LTP             # LTP time constant
        self.tau_postpre = tau_LTD             # LTD time constant
        self.A_prepost = A_LTP                 # LTP weight changing amplitude
        self.A_postpre = A_LTD                 # LTP weight changing amplitude
        self.pre_neuron = pre_neuron           # the previous neuron this synapse connects from
        self.type = syn_type                   # synapse type: excitatory or inhibitory
        self.delta_t = int_delta_t             # euler integration time step
        
        # the post neuron this synapse connects to (to be set after adding this synapse to the post neuron)
        self.post_neuron = None

        # define function to integrate the synapse conductance equation
        # the function f(g_tt) for the simplified membrane voltage equation: d g_tt / dtt = f(g_tt), g_tt += w_tt if spike
        # whose full form is: d g_tt / dtt = - g_tt/tau_syn + w_tt * Σ dirac(t - ts)
        # where ts is the spiking time of its pre-neuron
        self.func_g = lambda g_tt, tau_syn: -g_tt/tau_syn

        self.euler = Euler()                   # euler integrator
    def pre_spiking(self, time_step_sim):
        return self.tt - time_step_sim <= self.pre_neuron.last_spike < self.tt + time_step_sim
    def post_spiking(self, time_step_sim):
        return self.tt - time_step_sim <= self.post_neuron.last_spike < self.tt + time_step_sim
    def tick(self, time_step_sim):
        if self.pre_spiking(time_step_sim):
            # if pre-neuron is spiking, then add weight
            self.g_tt += self.w_tt

        # integrate the synapse conductance equation
        self.g_tt = self.euler.euler_integration(self.func_g, self.tau_syn, self.g_tt, self.tt, time_step_sim, self.delta_t)
        if self.type == "exc" and (self.pre_spiking(time_step_sim) or self.post_spiking(time_step_sim)):
            # if the pre neuron or post neuron is spiking, then apply STDP rules to update weights
            self.STDP()
        self.tt += time_step_sim
    def set_post_neuron(self, neuron):
        # set the post neuron of this synapse
        self.post_neuron = neuron
        if self.type == "exc":
            assert self.post_neuron.E_leak < self.E_syn
        elif self.type == "inh":
            assert self.post_neuron.E_leak > self.E_syn
    def STDP(self):
        # apply Spike-Timing Dependent Plasticity weight update
        Delta_t = self.pre_neuron.last_spike - self.post_neuron.last_spike
        if Delta_t > 0:
            Delta_w_e = self.A_postpre * np.exp(-Delta_t/self.tau_postpre)
        elif Delta_t < 0:
            Delta_w_e = self.A_prepost * np.exp(Delta_t/self.tau_prepost)
        else:
            Delta_w_e = 0
        self.w_tt += Delta_w_e
        self.w_tt = np.clip(self.w_tt, 0, w_max)

def generate_spike_trains():
    ###########################
    # create input spike trains
    ###########################
    
    # firing rates : 
    r1 = firing_rate_e
    r2 = firing_rate_e
    r3 = firing_rate_i
    r4 = firing_rate_i
    #### get correlated spike tains for excitatory input
    
    ### instantaneous correlations:
    spikes_e_corr = Correlated_Trains()
    [list_of_all_spike_trains1,list_of_all_spike_trains2] = spikes_e_corr.get_list_of_trains(c1,c2,firing_rate_e)
    
    ### jittered (exponential) correlations:
    #spikes_e_corr = CorrelatedJitter_Trains()
    #[list_of_all_spike_trains1,list_of_all_spike_trains2] = spikes_e_corr.get_list_of_trains(c1,c2,firing_rate_e,tau_c)
    
    spike_trains_complete_e = list_of_all_spike_trains1 + list_of_all_spike_trains2
    
    spikes_i = Poisson_Trains()
    [list_of_all_spike_trains1,list_of_all_spike_trains2] = spikes_i.get_list_of_trains(r3,r4)
    spike_trains_complete_i = list_of_all_spike_trains1 + list_of_all_spike_trains2
    return spike_trains_complete_e, spike_trains_complete_i


def create_neuron_synapse():
    spike_trains_complete_e, spike_trains_complete_i = generate_spike_trains()

    exc_syns = []
    exc_pre_neurons = []
    for i in range(numb_exc_syn):
        # create an input neuron with the manually generated spike trains
        exc_neuron = InputNeuron(t_0+time_step_sim, spike_trains_complete_e[i])
        exc_pre_neurons.append(exc_neuron)

        # create the exitatory synapse of this input neuron
        exc_syn = Synapse(t_0+time_step_sim, w_e, E_e, tau_e, exc_neuron, "exc")
        exc_syns.append(exc_syn)

    inh_syns = []
    inh_pre_neurons = []
    for i in range(numb_inh_syn):
        # create an input neuron with the manually generated spike trains
        inh_neuron = InputNeuron(t_0+time_step_sim, spike_trains_complete_i[i])
        inh_pre_neurons.append(inh_neuron)

        # create the inhibitory synapse of this input neuron
        inh_syn = Synapse(t_0+time_step_sim, w_i, E_i, tau_i, inh_neuron, "inh")
        inh_syns.append(inh_syn)

    # create ego neuron which accepts inputs from input neurons
    ego = Neuron(t_0+time_step_sim)
    ego_input_syns = exc_syns + inh_syns
    ego.add_synapses(ego_input_syns)

    all_syns = ego_input_syns
    all_neurons = [ego] + exc_pre_neurons + inh_pre_neurons
    return all_neurons, all_syns

def sim():
    all_neurons, all_syns = create_neuron_synapse()
    tt = t_0 + time_step_sim

    number_spikes = 0
    FR_vec = []

    w_e_storage = np.zeros((int(round((t_max-t_0)/time_step_sim))+1, numb_exc_syn))
    w_e_storage[0, :] = [syn.w_tt for syn in all_syns[:numb_exc_syn]]
    counter_storage = 1

    while tt <= t_max:
        for neuron in all_neurons:
            neuron.tick(time_step_sim)
        for syn in all_syns:
            syn.tick(time_step_sim)
        tt += time_step_sim

        # record the synapse weights
        w_e_storage[counter_storage,:] = [syn.w_tt for syn in all_syns[:numb_exc_syn]]
        counter_storage += 1

        # record the spike frequency
        if all_neurons[0].V_tt == V_reset:
            number_spikes += 1
        if tt%1000==0:
            FR_vec.append(number_spikes)
            number_spikes = 0
    fig, ax = plt.subplots()
    ax.plot(FR_vec)
    fig.savefig("firing rate.png")

    fig1, ax2 = plt.subplots()
    ax2.plot(range(int(round((t_max-t_0)/time_step_sim))+1),np.mean(w_e_storage[:,0:int(numb_exc_syn*0.5)],axis=1),lw=3,label='Corr : ' + str(c1),color='m')
    ax2.plot(range(int(round((t_max-t_0)/time_step_sim))+1),np.mean(w_e_storage[:,int(numb_exc_syn*0.5):numb_exc_syn],axis=1),lw=3,label='Corr : ' + str(c2),color='g')
    ax2.legend()
    ax2.plot(range(int(round((t_max-t_0)/time_step_sim))+1),w_e_storage[:,0:int(numb_exc_syn*0.5)],lw=0.5,label='Corr : ' + str(c1),color='m')
    ax2.plot(range(int(round((t_max-t_0)/time_step_sim))+1),w_e_storage[:,int(numb_exc_syn*0.5):numb_exc_syn],lw=0.5,label='Corr : ' + str(c2),color='g')
    ax2.set_xticks([0,t_max * 0.5, t_max])
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Syn. Weight')
    plt.tight_layout()
    plt.show()
    fig1.savefig('STDP_correl.png')

if __name__ == "__main__":
    sim()
