import numpy as np
import Parameters_Int_and_Fire
from Poisson_Spike_Trains import Poisson_Trains
from Correlated_Spike_Trains import Correlated_Trains
import matplotlib.pyplot as plt
import networkx as nx
import time
from multiprocessing import Pool
import ray
import uuid
ray.init(local_mode=False)

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
@ray.remote
class InputNeuron:
    def __init__(self, init_tt, spike_train):
        self.id = str(uuid.uuid4())
        self.tt = init_tt
        self.spike_train = spike_train # the prescribed spike train of this input neuron

        # the last time when this neuron spiked
        # this signal will be used for the conductance calculation of its synapses
        self.last_spike = init_tt

        # the next index of the spike in the spike train which is ready for spiking
        self.next_spike_idx = 0

        self.synapses_ids = []
    def get_id(self):
        return self.id
    def get_V_tt(self):
        return self.V_tt
    def get_last_spike(self):
        return self.last_spike
    def get_synapses_ids(self):
        return self.synapses_ids
    def tick(self, time_step_sim, g_tt, E_syn):
        while self.past_spike(self.next_spike_idx, self.tt, time_step_sim) and self.next_spike_idx < len(self.spike_train)-1:
            # if sim interval went past current spike, proceed to the next spike
            self.next_spike_idx += 1
        if self.cover_spike(self.next_spike_idx, self.tt, time_step_sim):
            # if the next spike is ready for spiking (covered in current simulation time interval)
            # then record it as the last spike time
            self.last_spike = self.spike_train[self.next_spike_idx]
        self.tt += time_step_sim
        return True
    def cover_spike(self, idx, tt, time_step_sim):
        return tt <= self.spike_train[idx] < tt + time_step_sim
    def past_spike(self, idx, tt, time_step_sim):
        return self.spike_train[idx] < tt
    def before_spike(self, idx, tt, time_step_sim):
        return tt + time_step_sim <= self.spike_train[idx]


@ray.remote
class Neuron:
    def __init__(self, init_tt, init_V_tt=-70, tau_mem=20, E_leak=-60, V_thresh=-50, V_reset=-70, int_delta_t=0.01):
        self.id = str(uuid.uuid4())
        self.V_tt = init_V_tt         # membrane voltage
        self.tt = init_tt             # current time
        self.tau_mem = tau_mem        # membrane time constant
        self.E_leak = E_leak          # reversal potential for the leak
        self.V_thresh = V_thresh      # membrane voltage threshold, V_tt will be reset to V_reset after reaching this threshold
        self.V_reset = V_reset        # the reset voltage
        self.delta_t = int_delta_t    # euler integration time step
        self.synapses_ids = []

        self.last_spike = init_tt     # the time when this neuron last spiked

        # define function to integrate the membrane voltage equation
        # the function f(V_tt) for the simplified membrane voltage equation: d V_tt / dtt = f(V_tt)
        # whose full form is: tau_mem * d V_tt / dtt = E_leak - V_tt + g_e * (E_e - V_tt) + g_i * (E_i - V_tt)
        self.func_V = lambda V_tt, syn_input_tt: (self.E_leak - V_tt + syn_input_tt)/self.tau_mem

        self.euler = Euler()          # euler integrator
    def get_id(self):
        return self.id
    def get_V_tt(self):
        return self.V_tt
    def get_last_spike(self):
        return self.last_spike
    def get_synapses_ids(self):
        return self.synapses_ids
    def add_synapses_ids(self, synapses_ids):
        self.synapses_ids += synapses_ids
    def tick(self, time_step_sim, g_tt, E_syn):
        # simulate the neuron for one step
        n_syns = len(g_tt)
        assert n_syns == len(E_syn)

        # sum all the synapse inputs
        syn_input_tt = np.sum(g_tt * (E_syn - self.V_tt))
        # integrate the membrane voltage equation
        V = self.euler.euler_integration(self.func_V, syn_input_tt, self.V_tt, self.tt, time_step_sim, self.delta_t)
        if V < self.V_thresh:
            self.V_tt = V
        else:
            self.V_tt = self.V_reset
            self.last_spike = self.tt
        self.tt += time_step_sim
        return True

@ray.remote
class Synapse:
    def __init__(self, init_tt, init_w_tt, E_syn, tau_syn, pre_neuron_id, post_neuron_id, syn_type, init_g_tt=0, w_max=40, tau_LTP=17, tau_LTD=34, A_LTP=0.02, A_LTD=-0.01, int_delta_t=0.01):
        self.id = str(uuid.uuid4())
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
        self.pre_neuron_id = pre_neuron_id     # the previous neuron this synapse connects from
        self.post_neuron_id = post_neuron_id   # the post neuron this synapse connects to

        self.type = syn_type                   # synapse type: excitatory or inhibitory

        self.delta_t = int_delta_t             # euler integration time step
        
        # define function to integrate the synapse conductance equation
        # the function f(g_tt) for the simplified membrane voltage equation: d g_tt / dtt = f(g_tt), g_tt += w_tt if spike
        # whose full form is: d g_tt / dtt = - g_tt/tau_syn + w_tt * Σ dirac(t - ts)
        # where ts is the spiking time of its pre-neuron
        self.func_g = lambda g_tt, tau_syn: -g_tt/tau_syn

        self.euler = Euler()                   # euler integrator
    def get_id(self):
        return self.id
    def get_g_tt(self):
        return self.g_tt
    def get_E_syn(self):
        return self.E_syn
    def get_w_tt(self):
        return self.w_tt
    def get_pre_neuron_id(self):
        return self.pre_neuron_id
    def get_post_neuron_id(self):
        return self.post_neuron_id
    def pre_spiking(self, time_step_sim, pre_spike):
        return self.tt - time_step_sim <= pre_spike < self.tt + time_step_sim
    def post_spiking(self, time_step_sim, post_spike):
        return self.tt - time_step_sim <= post_spike < self.tt + time_step_sim
    def tick(self, time_step_sim, pre_spike, post_spike):
        if self.pre_spiking(time_step_sim, pre_spike):
            # if pre-neuron is spiking, then add weight
            self.g_tt += self.w_tt

        # integrate the synapse conductance equation
        self.g_tt = self.euler.euler_integration(self.func_g, self.tau_syn, self.g_tt, self.tt, time_step_sim, self.delta_t)
        if self.type == "exc" and (self.pre_spiking(time_step_sim, pre_spike) or self.post_spiking(time_step_sim, post_spike)):
            # if the pre neuron or post neuron is spiking, then apply STDP rules to update weights
            self.STDP(pre_spike, post_spike)
        self.tt += time_step_sim
        return True
    def STDP(self, pre_spike, post_spike):
        # apply Spike-Timing Dependent Plasticity weight update
        Delta_t = pre_spike - post_spike
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



def create_neuron_synapse_networkx():
    n_hidden = 2
    n_hidden_syns = 1
    spike_trains_complete_e, spike_trains_complete_i = generate_spike_trains()
    hidden_neurons = [Neuron.remote(t_0+time_step_sim) for i in range(n_hidden)]
    hidden_neurons_ids = ray.get([neuron.get_id.remote() for neuron in hidden_neurons])

    G = nx.DiGraph()
    for i in range(n_hidden):
        neuron_id = hidden_neurons_ids[i]
        neuron = hidden_neurons[i]
        G.add_node(neuron_id, input=False, neuron=neuron)
    for i in range(numb_exc_syn):
        # create an input neuron with the manually generated spike trains
        pre_neuron = InputNeuron.remote(t_0+time_step_sim, spike_trains_complete_e[i])
        pre_neuron_id = ray.get(pre_neuron.get_id.remote())

        # create the exitatory synapse from this input neuron to a random hidden neuron
        post_neuron = np.random.choice(hidden_neurons)
        post_neuron_id = ray.get(post_neuron.get_id.remote())

        exc_syn = Synapse.remote(t_0+time_step_sim, w_e, E_e, tau_e, pre_neuron_id, post_neuron_id, "exc")
        exc_syn_id = ray.get(exc_syn.get_id.remote())
        post_neuron.add_synapses_ids.remote([exc_syn_id])

        G.add_node(pre_neuron_id, input=True, neuron=pre_neuron)
        G.add_edge(pre_neuron_id, post_neuron_id, syn_id=exc_syn_id, syn=exc_syn)

    for i in range(numb_inh_syn):
        # create an input neuron with the manually generated spike trains
        pre_neuron = InputNeuron.remote(t_0+time_step_sim, spike_trains_complete_i[i])
        pre_neuron_id = ray.get(pre_neuron.get_id.remote())

        # create the inhibitory synapse from this input neuron to a random hidden neuron
        post_neuron = np.random.choice(hidden_neurons)
        post_neuron_id = ray.get(post_neuron.get_id.remote())

        inh_syn = Synapse.remote(t_0+time_step_sim, w_i, E_i, tau_i, pre_neuron_id, post_neuron_id, "inh")
        inh_syn_id = ray.get(inh_syn.get_id.remote())
        post_neuron.add_synapses_ids.remote([inh_syn_id])

        G.add_node(pre_neuron_id, input=True, neuron=pre_neuron)
        G.add_edge(pre_neuron_id, post_neuron_id, syn_id=inh_syn_id, syn=inh_syn)
    all_neurons_ids = list(G.nodes)
    all_neurons = [G.nodes[nid]['neuron'] for nid in all_neurons_ids]
    hidden_neurons_ids = [nid for nid in G.nodes if G.nodes[nid]['input'] == False]
    hidden_neurons = [G.nodes[nid]['neuron'] for nid in hidden_neurons_ids]
    nid2neuron = dict(zip(all_neurons_ids, all_neurons))

    for i in range(n_hidden_syns):
        pre_neuron_id = np.random.choice(hidden_neurons_ids)
        post_neuron_id = np.random.choice(hidden_neurons_ids)
        while G.has_edge(pre_neuron_id, post_neuron_id) or G.has_edge(post_neuron_id, pre_neuron_id) or pre_neuron_id == post_neuron_id:
            pre_neuron_id = np.random.choice(hidden_neurons_ids)
            post_neuron_id = np.random.choice(hidden_neurons_ids)
        if np.random.rand() < 0.8:
            hidden_syn = Synapse.remote(t_0+time_step_sim, w_e, E_e, tau_e, pre_neuron_id, post_neuron_id, "exc")
        else:
            hidden_syn = Synapse.remote(t_0+time_step_sim, w_i, E_i, tau_i, pre_neuron_id, post_neuron_id, "inh")
        hidden_syn_id = ray.get(hidden_syn.get_id.remote())
        nid2neuron[post_neuron_id].add_synapses_ids.remote([hidden_syn_id])
        G.add_edge(pre_neuron_id, post_neuron_id, syn_id=hidden_syn_id, syn=hidden_syn)
    layout = nx.spring_layout(G)
    nx.draw_networkx(G, pos=layout, arrows=True, node_color=['r' if G.nodes[u]['input'] else 'k' for u in G.nodes], node_size=50, with_labels=False)
    plt.savefig("network_topo.png")
    plt.close()
    return G

def sim_networkx():
    G = create_neuron_synapse_networkx()
    all_neurons_ids = list(G.nodes)
    all_neurons = [G.nodes[nid]['neuron'] for nid in all_neurons_ids]
    nid2neuron = dict(zip(all_neurons_ids, all_neurons))

    all_syns_ids = [G.edges[e]['syn_id'] for e in G.edges]
    all_syns = [G.edges[e]['syn'] for e in G.edges]
    sid2syn = dict(zip(all_syns_ids, all_syns))

    hidden_neurons_ids = [nid for nid in G.nodes if G.nodes[nid]['input'] == False]
    hidden_neurons = [G.nodes[nid]['neuron'] for nid in hidden_neurons_ids]

    n_neurons = len(all_neurons_ids)
    n_hidden = len(hidden_neurons_ids)
    n_syns = len(all_syns_ids)

    all_neuron_in_syns = []
    for i in range(n_neurons):
        neuron = all_neurons[i]

        synapses_ids = ray.get(neuron.get_synapses_ids.remote())
        synapses = [sid2syn[sid] for sid in synapses_ids]

        all_neuron_in_syns.append(synapses)

    all_syn_pre_neuron = []
    all_syn_post_neuron = []
    for i in range(n_syns):
        syn = all_syns[i]

        pre_neuron_id  = ray.get(syn.get_pre_neuron_id.remote())
        pre_neuron = nid2neuron[pre_neuron_id]

        post_neuron_id  = ray.get(syn.get_post_neuron_id.remote())
        post_neuron = nid2neuron[post_neuron_id]

        all_syn_pre_neuron.append(pre_neuron)
        all_syn_post_neuron.append(post_neuron)
    tt = t_0 + time_step_sim

    number_spikes = [0] * n_hidden
    FR_vec = [[] for i in range(n_hidden)]

    w_e_storage = np.zeros((int(round((t_max-t_0)/time_step_sim))+1, n_syns))
    w_e_storage[0, :] = [ray.get(sid2syn[sid].get_w_tt.remote()) for sid in all_syns_ids]
    counter_storage = 1

    while tt <= t_max:
        g_tts = []
        tik = time.time()
        for i in range(n_neurons):
            g_tt = ray.get([syn.get_g_tt.remote() for syn in all_neuron_in_syns[i]])
            g_tts.append(g_tt)
        print("get g_tt:", time.time() - tik)

        E_syns = []
        tik = time.time()
        for i in range(n_neurons):
            E_syn = ray.get([syn.get_E_syn.remote() for syn in all_neuron_in_syns[i]])
            E_syns.append(E_syn)
        print("get Esyn:", time.time() - tik)

        results = []
        tik = time.time()
        for i in range(n_neurons):
            g_tt = ray.put(np.array(g_tts[i]))
            E_syn = ray.put(np.array(E_syns[i]))
            r = all_neurons[i].tick.remote(time_step_sim, g_tt, E_syn)
            results.append(r)
        ray.get(results)
        print("sim neurons:",time.time() - tik)

        results = []
        tik = time.time()
        for i in range(n_syns):
            pre_neuron = all_syn_pre_neuron[i]
            post_neuron = all_syn_post_neuron[i]
            pre_spike = ray.get(pre_neuron.get_last_spike.remote())
            post_spike = ray.get(post_neuron.get_last_spike.remote())
            r = all_syns[i].tick.remote(time_step_sim, pre_spike, post_spike)
        ray.get(results)
        print("sim syns:",time.time() - tik)

        tt += time_step_sim

        # record the synapse weights
        w_e_storage[counter_storage,:] = ray.get([syn.get_w_tt.remote() for syn in all_syns])
        counter_storage += 1

        # record the spike frequency
        for i in range(n_hidden):
            hidden = hidden_neurons[i]
            if ray.get(hidden.get_V_tt.remote()) == V_reset:
                number_spikes[i] += 1
            if tt%1000==0:
                FR_vec[i].append(number_spikes[i])
                number_spikes[i] = 0

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

if __name__ == "__main__":
    tik = time.time()
    sim_networkx()
    tok = time.time()
    print(tok - tik)
