import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import numpy as np
import Parameters_Int_and_Fire
from Poisson_Spike_Trains import Poisson_Trains
from Correlated_Spike_Trains import Correlated_Trains
import matplotlib.pyplot as plt
import networkx as nx
import time
from typing import Type

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

n=int(round(time_step_sim/delta_t)) # calculate the number of steps
def euler_integration(f, arg_for_f, y_0, t_0, delta_t):
    for step in range(1,n+1):
        m=f(y_0, arg_for_f)
        y_1=y_0+delta_t*m
        t_1=t_0+delta_t
        t_0=t_1
        y_0=y_1
    return y_0

@register_pytree_node_class
class Neuron:
    def __init__(self, init_tt, init_last_spike, init_V_tt=-70, tau_mem=20, E_leak=-60, V_thresh=-50, V_reset=-70, int_delta_t=0.01):
        self.tt = init_tt
        self.V_tt = init_V_tt         # membrane voltage
        self.tau_mem = tau_mem        # membrane time constant
        self.E_leak = E_leak          # reversal potential for the leak
        self.V_thresh = V_thresh      # membrane voltage threshold, V_tt will be reset to V_reset after reaching this threshold
        self.V_reset = V_reset        # the reset voltage
        self.delta_t = int_delta_t    # euler integration time step

        self.last_spike = init_last_spike     # the time when this neuron last spiked

        # define function to integrate the membrane voltage equation
        # the function f(V_tt) for the simplified membrane voltage equation: d V_tt / dtt = f(V_tt)
        # whose full form is: tau_mem * d V_tt / dtt = E_leak - V_tt + g_e * (E_e - V_tt) + g_i * (E_i - V_tt)
        self.func_V = lambda V_tt, syn_input_tt: (self.E_leak - V_tt + syn_input_tt)/self.tau_mem

    def tick(self, time_step_sim, synapses, in_syns_logits):
        # simulate the neuron for one step

        # sum all the synapse inputs
        syn_input_tt = 0
        g_tts = jnp.asarray([syn.g_tt for syn in synapses])
        E_syns = jnp.asarray([syn.E_syn for syn in synapses])
        syn_input_tt = jnp.sum(g_tts * (E_syns - self.V_tt) * in_syns_logits)
        # integrate the membrane voltage equation
        V = euler_integration(self.func_V, syn_input_tt, self.V_tt, self.tt, self.delta_t)
        V_tt = jnp.where(V < self.V_thresh, V, self.V_reset)
        last_spike = jnp.where(V < self.V_thresh, self.last_spike, self.tt)
        return Neuron(self.tt + time_step_sim, last_spike, V_tt, self.tau_mem, self.E_leak, self.V_thresh, self.V_reset, self.delta_t)

    def tree_flatten(self):
        children = (self.tt, self.last_spike, self.V_tt, self.tau_mem, self.E_leak, self.V_thresh, self.V_reset, self.delta_t)
        aux_data = None
        return (children, aux_data)
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@register_pytree_node_class
class InputNeuron:
    def __init__(self, init_tt, last_spike, next_spike_idx, spike_train):
        self.tt = init_tt
        self.spike_train = spike_train # the prescribed spike train of this input neuron

        # the last time when this neuron spiked
        # this signal will be used for the conductance calculation of its synapses
        self.last_spike = last_spike

        # the next index of the spike in the spike train which is ready for spiking
        self.next_spike_idx = next_spike_idx
    def tick(self, time_step_sim):
        last_spike = self.last_spike
        next_spike_idx = self.next_spike_idx
        while self.past_spike(self.next_spike_idx, self.tt, time_step_sim) and self.next_spike_idx < len(self.spike_train)-1:
            # if sim interval went past current spike, proceed to the next spike
            next_spike_idx += 1
        if self.cover_spike(self.next_spike_idx, self.tt, time_step_sim):
            # if the next spike is ready for spiking (covered in current simulation time interval)
            # then record it as the last spike time
            last_spike = self.spike_train[self.next_spike_idx]
        return InputNeuron(self.tt + time_step_sim, last_spike, next_spike_idx, self.spike_train)
    def cover_spike(self, idx, tt, time_step_sim):
        return tt <= self.spike_train[idx] < tt + time_step_sim
    def past_spike(self, idx, tt, time_step_sim):
        return self.spike_train[idx] < tt
    def before_spike(self, idx, tt, time_step_sim):
        return tt + time_step_sim <= self.spike_train[idx]

    def tree_flatten(self):
        children = (self.tt, self.last_spike, self.next_spike_idx, self.spike_train)
        aux_data = None
        return (children, aux_data)
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@register_pytree_node_class
class Synapse:
    def __init__(self, init_tt, init_w_tt, E_syn, tau_syn, pre_neuron_idx, post_neuron_idx, syn_type, init_g_tt=0, w_max=40, tau_LTP=17, tau_LTD=34, A_LTP=0.02, A_LTD=-0.01, int_delta_t=0.01):
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
        self.pre_neuron_idx = pre_neuron_idx           # the previous neuron this synapse connects from
        self.post_neuron_idx = post_neuron_idx         # the post neuron this synapse connects to
        self.type = syn_type                   # synapse type: excitatory or inhibitory

        self.delta_t = int_delta_t             # euler integration time step
        
        # define function to integrate the synapse conductance equation
        # the function f(g_tt) for the simplified membrane voltage equation: d g_tt / dtt = f(g_tt), g_tt += w_tt if spike
        # whose full form is: d g_tt / dtt = - g_tt/tau_syn + w_tt * Σ dirac(t - ts)
        # where ts is the spiking time of its pre-neuron
        self.func_g = lambda g_tt, tau_syn: -g_tt/tau_syn

    def tick(self, time_step_sim, neurons):
        g_tt = self.g_tt
        last_spikes = jnp.asarray([n.last_spike for n in neurons])
        pre_spike = jnp.take(last_spikes, self.pre_neuron_idx)
        post_spike = jnp.take(last_spikes, self.post_neuron_idx)

        # logic AND in jnp
        pre_spiked1 = jnp.where(self.tt - time_step_sim <= pre_spike, 1, 0)
        pre_spiked = jnp.where(pre_spike < self.tt + time_step_sim, pre_spiked1, 0)
        post_spiked1 = jnp.where(self.tt - time_step_sim <= post_spike, 1, 0)
        post_spiked = jnp.where(post_spike < self.tt + time_step_sim, post_spiked1, 0)
        # if pre-neuron is spiking, then add weight
        g_tt += self.w_tt * pre_spiked

        # integrate the synapse conductance equation
        g_tt = euler_integration(self.func_g, self.tau_syn, g_tt, self.tt, self.delta_t)
        # if the pre neuron or post neuron is spiking, then apply STDP rules to update weights
        w_STDP = self.STDP(pre_spike, post_spike)
        # logic OR in jnp
        w_tt = jnp.where(pre_spiked, w_STDP, self.w_tt)
        w_tt = jnp.where(post_spiked, w_STDP, w_tt)
        return Synapse(self.tt + time_step_sim, w_tt, self.E_syn, self.tau_syn, self.pre_neuron_idx, self.post_neuron_idx, self.type, g_tt, self.w_max, self.tau_prepost, self.tau_postpre, self.A_prepost, self.A_postpre, self.delta_t)

    def STDP(self, pre_spike, post_spike):
        # apply Spike-Timing Dependent Plasticity weight update
        Delta_t = pre_spike - post_spike
        Delta_w_e = jnp.where(Delta_t > 0, self.A_postpre * jnp.exp(-Delta_t/self.tau_postpre), self.A_prepost * jnp.exp(Delta_t/self.tau_prepost))
        w_tt = self.w_tt + Delta_w_e
        w_tt = jnp.clip(w_tt, 0, w_max)
        return w_tt

    def tree_flatten(self):
        children = (self.tt, self.w_tt, self.E_syn, self.tau_syn, self.pre_neuron_idx, self.post_neuron_idx, self.type, self.g_tt, self.w_max, self.tau_prepost, self.tau_postpre, self.A_prepost, self.A_postpre, self.delta_t)
        aux_data = None
        return (children, aux_data)
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
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
def one_hot(i, n):
    a = np.zeros(n)
    a[i] =1
    return a
def create_neuron_synapse_networkx():
    n_hidden = 20
    n_input = numb_exc_syn + numb_inh_syn
    n_neurons = n_hidden + n_input
    spike_trains_complete_e, spike_trains_complete_i = generate_spike_trains()

    G = nx.gnp_random_graph(n_neurons, 0.4, directed=True)
    assert len(G.edges) > n_input
    neurons = []
    input_neurons = []
    hidden_neurons = []
    for i in range(n_hidden):
        neuron = Neuron(t_0+time_step_sim, t_0+time_step_sim)
        neurons.append(neuron)
        hidden_neurons.append(neuron)
    for i in range(numb_exc_syn):
        neuron = InputNeuron(t_0+time_step_sim, t_0+time_step_sim, 0, jnp.asarray(spike_trains_complete_e[i]))
        neurons.append(neuron)
        input_neurons.append(neuron)
    for i in range(numb_inh_syn):
        neuron = InputNeuron(t_0+time_step_sim, t_0+time_step_sim, 0, jnp.asarray(spike_trains_complete_i[i]))
        neurons.append(neuron)
        input_neurons.append(neuron)
    syns = []
    for pre_neuron_idx, post_neuron_idx in G.edges:
        if pre_neuron_idx < n_hidden + numb_exc_syn:
            syns.append(Synapse(t_0+time_step_sim, w_e, E_e, tau_e, pre_neuron_idx, post_neuron_idx, 1))
        else:
            syns.append(Synapse(t_0+time_step_sim, w_e, E_e, tau_e, pre_neuron_idx, post_neuron_idx, 0))
    return tuple(neurons), tuple(syns), tuple(hidden_neurons), tuple(input_neurons), nx.to_numpy_matrix(G)


def sim_jit():
    neurons, syns, hidden_neurons, input_neurons, adj_matrix = create_neuron_synapse_networkx()
    def step(tt, time_step_sim, hidden_neurons, syns, input_neurons):
        print("stepping neurons")
        hidden_neurons_ = tuple(hidden_neurons[i].tick(time_step_sim, syns, adj_matrix[:, i]) for i in range(len(hidden_neurons)))
        print("stepping syns")
        syns_ = tuple(s.tick(time_step_sim, hidden_neurons + input_neurons) for s in syns)
        return (tt + time_step_sim, hidden_neurons_, syns_)

    def update_input(time_step_sim, input_neurons):
        return tuple(n.tick(time_step_sim) for n in input_neurons)

    n_syns = len(syns)
    n_hidden = len(hidden_neurons)
    tt = t_0 + time_step_sim

    number_spikes = [0] * n_hidden
    FR_vec = [[] for i in range(n_hidden)]

    w_e_storage = np.zeros((int(round((t_max-t_0)/time_step_sim))+1, n_syns))
    w_e_storage[0, :] = [syn.w_tt for syn in syns]
    counter_storage = 1

    step_jit = jax.jit(step, static_argnums=1) # static argnums could be removed?
    while tt <= t_max:
        print("starting update input neurons")
        input_neurons = update_input(time_step_sim, input_neurons)
        print("starting step neurons and syns")
        tt, hidden_neurons, syns = step_jit(tt, time_step_sim, hidden_neurons, syns, input_neurons)
        # record the synapse weights
        w_e_storage[counter_storage,:] = [syn.w_tt for syn in syns]
        counter_storage += 1

        # record the spike frequency
        for i in range(n_hidden):
            hidden = hidden_neurons[i]
            if hidden.V_tt == V_reset:
                number_spikes[i] += 1
            if tt%1000==0:
                FR_vec[i].append(number_spikes[i])
                number_spikes[i] = 0
        print(tt)
    fig, ax = plt.subplots()
    ax.plot(FR_vec)
    fig.savefig("firing_rate_nx.png")

    fig1, ax2 = plt.subplots()
    ax2.plot(range(int(round((t_max-t_0)/time_step_sim))+1),w_e_storage)
    ax2.set_xticks([0,t_max * 0.5, t_max])
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Syn. Weight')
    plt.tight_layout()
    fig1.savefig('STDP_correl_nx.png')
sim_jit()
