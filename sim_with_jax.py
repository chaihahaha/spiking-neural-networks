import jax
import jax.numpy as jnp
from jax.experimental import sparse
import numpy as np
import Parameters_Int_and_Fire
from Poisson_Spike_Trains import Poisson_Trains
from Correlated_Spike_Trains import Correlated_Trains
import matplotlib.pyplot as plt
import networkx as nx
import time

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
        while self.past_spike(self.next_spike_idx, self.tt, time_step_sim) and self.next_spike_idx < len(self.spike_train)-1:
            # if sim interval went past current spike, proceed to the next spike
            self.next_spike_idx += 1
        if self.cover_spike(self.next_spike_idx, self.tt, time_step_sim):
            # if the next spike is ready for spiking (covered in current simulation time interval)
            # then record it as the last spike time
            self.last_spike = self.spike_train[self.next_spike_idx]
        self.tt += time_step_sim
        return
    def cover_spike(self, idx, tt, time_step_sim):
        return tt <= self.spike_train[idx] < tt + time_step_sim
    def past_spike(self, idx, tt, time_step_sim):
        return self.spike_train[idx] < tt
    def before_spike(self, idx, tt, time_step_sim):
        return tt + time_step_sim <= self.spike_train[idx]

E_e=0.
E_i=-80.
tau_e=3.
tau_i=5.
tau_mem=20.
E_leak=-60.
V_thresh=-50.
V_reset=-70.
w_max=40.
tau_prepost=17.
tau_postpre=34.
A_prepost=0.02
A_postpre=-0.01
delta_t=0.01

def neurons_tick(tt, neurons_last_spike, V_tt, time_step_sim, g_tts, E_syns, neurons_in_syns_logits):
    # S is #synapses, I is #input neurons, H is #hidden neurons
    # neurons_last_spike : [ I + H, 1 ]
    # V_tt : [ I + H, 1 ]
    # time_step_sim : [0]
    # g_tts : [ S, 1 ]
    # E_syns : [ S, 1 ]
    # in_syns_logits : [ I + H, S ]

    # simulate the neuron for one step
    g_tts = jnp.reshape(g_tts, [1, -1])
    E_syns = jnp.reshape(E_syns, [1, -1])

    # sum all the synapse inputs
    syn_input_tt = jnp.sum(g_tts * (E_syns - V_tt) * neurons_in_syns_logits, axis=1, keepdims=True)  # [ I + H, 1]
    # integrate the membrane voltage equation
    V = (V_tt - E_leak - syn_input_tt) * jnp.exp(-time_step_sim / tau_mem) + E_leak + syn_input_tt
    V_tt = jnp.where(V < V_thresh, V, V_reset)
    neurons_last_spike = jnp.where(V < V_thresh, neurons_last_spike, tt)
    return tt + time_step_sim, neurons_last_spike, V_tt

def synapses_tick(tt, g_tts, w_tts, time_step_sim, neurons_last_spike, pre_neurons_logits, post_neurons_logits, taus_syn):
    # S is #synapses, I is #input neurons, H is #hidden neurons
    # g_tts : [ S , 1 ]
    # w_tts : [ S , 1 ]
    # time_step_sim : [0]
    # neurons_last_spike : [ I + H, 1 ]
    # pre_neurons_logits : [ S, I + H ]
    # post_neurons_logits : [ S, I + H ]
    # taus_syn : [ S, 1 ]

    pre_spike = pre_neurons_logits @ neurons_last_spike   # [ S, 1 ]
    post_spike = post_neurons_logits @ neurons_last_spike # [ S, 1 ]

    pre_spiked = jnp.abs(pre_spike - tt) < time_step_sim     # [ S, 1 ]
    post_spiked = jnp.abs(post_spike - tt) < time_step_sim  # [ S, 1 ]
    # if pre-neuron is spiking, then add weight
    g_tts += w_tts * pre_spiked

    # integrate the synapse conductance equation
    g_tts = g_tts * jnp.exp(- time_step_sim / taus_syn)
    # if the pre neuron or post neuron is spiking, then apply STDP rules to update weights
    w_STDP = STDP(pre_spike, post_spike, w_tts)
    # logic OR in jnp
    pre_or_post_spiked = 1 - (1 - pre_spiked) * (1 - post_spiked)
    w_tts = jnp.where(pre_or_post_spiked, w_STDP, w_tts)
    return tt + time_step_sim, w_tts, g_tts

def STDP(pre_spike, post_spike, w_tt):
    # apply Spike-Timing Dependent Plasticity weight update
    Delta_t = pre_spike - post_spike
    Delta_w_e = jnp.where(Delta_t > 0, jnp.where(Delta_t == 0, jnp.zeros_like(Delta_t), A_postpre * jnp.exp(-Delta_t/tau_postpre)), A_prepost * jnp.exp(Delta_t/tau_prepost))
    w_tt = w_tt + Delta_w_e
    w_tt = jnp.clip(w_tt, 0, w_max)
    return w_tt

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
def create_neuron_synapse_networkx():
    n_hidden = 10
    n_input = numb_exc_syn + numb_inh_syn
    n_neurons = n_input + n_hidden
    spike_trains_complete_e, spike_trains_complete_i = generate_spike_trains()

    G = nx.gnp_random_graph(n_neurons, 0.05, directed=True)
    for i_input in range(n_input):
        if not G[i_input]:
            i_hidden = np.random.choice(np.arange(n_input, n_neurons))
            G.add_edge(i_input, i_hidden)
    n_synapses = len(G.edges)
    synapses = list(G.edges)
    syn_idx = { synapses[i]: i for i in range(len(synapses)) }

    print("n neurons:",len(G.nodes))
    print("n syns:",len(G.edges))
    assert len(G.edges) > n_input

    neurons_last_spike = np.zeros([n_neurons, 1], dtype=float)
    V_tt = V_reset * np.ones([n_neurons, 1], dtype=float)
    neurons_in_syns_logits = np.zeros([n_neurons, n_synapses], dtype=int)

    g_tts = np.zeros([n_synapses, 1], dtype=float)
    w_tts = np.zeros([n_synapses, 1], dtype=float)
    pre_neurons_logits = np.zeros([n_synapses, n_neurons], dtype=int)
    post_neurons_logits = np.zeros([n_synapses, n_neurons], dtype=int)
    E_syns = np.zeros([n_synapses, 1], dtype=float)
    taus_syn = np.zeros([n_synapses, 1], dtype=float)

    input_neurons = []
    for i in range(numb_exc_syn):
        neuron = InputNeuron(t_0+time_step_sim, t_0+time_step_sim, 0, jnp.asarray(spike_trains_complete_e[i], dtype=float))
        input_neurons.append(neuron)
    for i in range(numb_inh_syn):
        neuron = InputNeuron(t_0+time_step_sim, t_0+time_step_sim, 0, jnp.asarray(spike_trains_complete_i[i], dtype=float))
        input_neurons.append(neuron)
    for pre_neuron_idx, post_neuron_idx in G.edges:
        syn = (pre_neuron_idx, post_neuron_idx)
        si = syn_idx[syn]
        pre_neurons_logits[si, pre_neuron_idx] = 1
        post_neurons_logits[si, post_neuron_idx] = 1
        if pre_neuron_idx < numb_exc_syn:
            w_tts[si, 0] = w_e
            E_syns[si, 0] = E_e
            taus_syn[si, 0] = tau_e
        elif pre_neuron_idx < n_input:
            w_tts[si, 0] = w_i
            E_syns[si, 0] = E_i
            taus_syn[si, 0] = tau_i
        else:
            # set synapses not from input neurons
            if np.random.rand() < 0.8:
                w_tts[si, 0] = w_e
                E_syns[si, 0] = E_e
                taus_syn[si, 0] = tau_e
            else:
                w_tts[si, 0] = w_i
                E_syns[si, 0] = E_i
                taus_syn[si, 0] = tau_i

    neurons_in_syns_logits = np.zeros([n_neurons, len(G.edges)])
    for i in range(len(G.nodes)):
        for j in G.predecessors(i):
            neurons_in_syns_logits[i, syn_idx[(j, i)]] = 1
    #layout = nx.spring_layout(G)
    #nx.draw_networkx(G, pos=layout, arrows=True, node_color=['r' if i>n_hidden else 'k' for i in range(len(G.nodes))], node_size=50, with_labels=False)
    #plt.savefig("network_topo.png")
    #plt.close()
    pre_neurons_logits = sparse.BCOO.fromdense(pre_neurons_logits)
    post_neurons_logits = sparse.BCOO.fromdense(post_neurons_logits)
    #neurons_in_syns_logits = sparse.BCOO.fromdense(neurons_in_syns_logits)
    return neurons_last_spike, V_tt, neurons_in_syns_logits, g_tts, w_tts, pre_neurons_logits, post_neurons_logits, E_syns, taus_syn, input_neurons, n_neurons, n_synapses

# not jax, avoid pytree copies
def update_input(time_step_sim, input_neurons, neurons_last_spike):
    neurons_last_spike = np.asarray(neurons_last_spike).copy()
    for i in range(len(input_neurons)):
        neuron = input_neurons[i]
        neuron.tick(time_step_sim)
        neurons_last_spike[i, 0] = neuron.last_spike
    return neurons_last_spike

def sim_jit():
    neurons_last_spike, V_tt, neurons_in_syns_logits, g_tts, w_tts, pre_neurons_logits, post_neurons_logits, E_syns, taus_syn, input_neurons, n_neurons, n_synapses = create_neuron_synapse_networkx()
    def step(tt, neurons_last_spike, V_tt, g_tts, w_tts):
        _, neurons_last_spike, V_tt = neurons_tick(tt, neurons_last_spike, V_tt, time_step_sim, g_tts, E_syns, neurons_in_syns_logits)
        _, w_tts, g_tts = synapses_tick(tt, g_tts, w_tts, time_step_sim, neurons_last_spike, pre_neurons_logits, post_neurons_logits, taus_syn)
        return (tt + time_step_sim, neurons_last_spike, V_tt, g_tts, w_tts)


    tt = t_0 + time_step_sim

    n_input = len(input_neurons)
    n_hidden = n_neurons - n_input
    number_spikes = [0] * n_hidden
    FR_vec = [[] for i in range(n_hidden)]

    w_e_storage = np.zeros((int(round((t_max-t_0)/time_step_sim))+1, n_synapses))
    w_e_storage[0, :] = np.reshape(w_tts, -1)
    counter_storage = 1

    step_jit = jax.jit(step) # static argnums could be removed?
    start_time = time.time()
    while tt <= t_max:
        tik = time.time()
        neurons_last_spike = update_input(time_step_sim, input_neurons, neurons_last_spike)  # when using GPU, this is slow (cost 10 seconds)

        tt, neurons_last_spike, V_tt, g_tts, w_tts = step_jit(tt, neurons_last_spike, V_tt, g_tts, w_tts)
        #print("V_tt", jnp.reshape(V_tt, -1))
        #print("g_tts", g_tts)
        #print("w_tts", jnp.reshape(w_tts, -1))
        # record the synapse weights
        w_e_storage[counter_storage,:] = np.reshape(w_tts, -1)
        counter_storage += 1

        # record the spike frequency
        for i in range(n_input, n_neurons):
            i_hidden = i - n_input
            if V_tt[i, 0] == V_reset:
                number_spikes[i_hidden] += 1
            if tt%1000==0:
                FR_vec[i_hidden].append(number_spikes[i_hidden])
                number_spikes[i_hidden] = 0
        print(time.time() - tik)
    print("#neuron:", n_neurons,"#syn:",  n_synapses)
    print("total time:", time.time() - start_time)
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
