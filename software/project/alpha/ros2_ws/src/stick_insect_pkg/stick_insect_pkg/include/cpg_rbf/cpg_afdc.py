import numpy as np

class CPG_AFDC:
    def __init__(self,
                 o0_init    = 0.2, 
                 o1_init    = 0.2, 
                 o2_init    = 0.2, 
                 phi_init   = 0.01 * 2 * np.pi, 
                 _alpha     = 1.01, 
                 lrate      = 1.3):
        
        self.out0_t = o0_init
        self.out1_t = o1_init
        self.out2_t = o2_init
        self.phi    =  phi_init

        self.alpha = _alpha
        self.w20_t = 0              # w -> weight
        self.w02_t = 1              # Default = 1
        self.w2p_t = 0.03

        self.hebbian_learning_rate = lrate 
        
        # Scaling factors
        factor   = 1
        self.A02 = 1 * factor
        self.A20 = 1 * factor
        self.A2p = 1
        self.B02 = 0.01 * factor
        self.B20 = 0.01 * factor
        self.B2p = 0.01 * factor
        
        
        self.w00 = self.alpha*np.cos(self.phi)
        self.w01 = self.alpha*np.sin(self.phi)
        self.w10 = self.alpha*(-np.sin(self.phi))
        self.w11 = self.alpha*np.cos(self.phi)
        

        self.discretize_count = 0
        self.discretize_factor = 10

        self.out0       = []
        self.out1       = []
        self.outFreq    = []

        self.w20_t1 = 0.0
        self.w02_t1 = 0.0

    # Only CPG
    def update_cpg_so2(self, phi):
        self.out0_t1 = np.tanh(self.w00*self.out0_t + self.w01*self.out1_t)
        self.out1_t1 = np.tanh(self.w10*self.out0_t + self.w11*self.out1_t)

        self.update_cpg_weights_with_phi(phi)

        # save for next iteration
        self.out0_t = self.out0_t1
        self.out1_t = self.out1_t1
        self.w20_t = self.w20_t1
        self.w02_t = self.w02_t1
        
    def update_cpg_weights_with_phi(self, phi):
        self.phi = phi
        # update cpg weight 
        self.w00 = self.alpha * np.cos(self.phi)
        self.w01 = self.alpha * np.sin(self.phi) 
        self.w10 = self.alpha * (-np.sin(self.phi)) 
        self.w11 = self.alpha * np.cos(self.phi)  

    # CPG-AFDC
    def update_adaptive_cpg_with_synaptic_plasticity(self, perturbation):
        # Adaptive CPG with Synaptic Plasticity (H0, H1, H2) in eq.(2)
        self.out0_t1 = np.tanh(self.w00*self.out0_t + self.w01*self.out1_t + self.w02_t*self.out2_t)
        self.out1_t1 = np.tanh(self.w10*self.out0_t + self.w11*self.out1_t)
        self.out2_t1 = np.tanh(self.w20_t*self.out0_t + self.w2p_t*(perturbation))  

        self.update_cpg_weights_with_learning_phi()    
        self.update_sensory_feedback_neuron_weights(perturbation)

        # save for next iteration
        self.out0_t = self.out0_t1
        self.out1_t = self.out1_t1
        self.out2_t = self.out2_t1  
        self.w20_t = self.w20_t1
        self.w02_t = self.w02_t1
        self.w2p_t = self.w2p_t1
        
    def update_cpg_weights_with_learning_phi(self):

        # update cpg phi with "Adaptation throgh Fast Dynamical Coupling (AFDC)" in eq.(4)
        self.delta_phi = self.hebbian_learning_rate*(self.w02_t*self.out2_t*self.w01*self.out1_t)    # Hebbian learning rule
        self.phi = self.phi + self.delta_phi                                            # Stocastic Gradient Descent (new = old + (lrate*error))

        # update cpg weight 
        self.w00 = self.alpha * np.cos(self.phi)
        self.w01 = self.alpha * np.sin(self.phi) 
        self.w10 = self.alpha * (-np.sin(self.phi)) 
        self.w11 = self.alpha * np.cos(self.phi)  

    def update_sensory_feedback_neuron_weights(self, perturbation):
        
        # initial predefined relaxation weight value
        w20_init = 0
        w02_init = 1
        w2p_init = 0.03 

        # update sensory feedback neuron weights in eq.(5)
        self.delta_w20 = - self.A20*self.out2_t*self.out0_t  - self.B20*(self.w20_t - w20_init)
        self.delta_w02 = - self.A02*self.out0_t*self.out2_t  - self.B02*(self.w02_t - w02_init)
        self.delta_w2p =   self.A2p*self.out2_t*perturbation - self.B2p*(self.w2p_t - w2p_init)

        self.w20_t1 = self.w20_t + self.delta_w20
        self.w02_t1 = self.w02_t + self.delta_w02
        self.w2p_t1 = self.w2p_t + self.delta_w2p








    # API 
    def update_cpg_with_discretize_factor(self, set_fcpg, discretize_factor):

        # freq.[Hz] = f_cpg[cpg_cycles per iteration] * 1/discretize_factor[iterations per program step] * update_rate[program step per second]

        self.discretize_factor = discretize_factor
        phi = set_fcpg * 2 * np.pi                                  
        if self.discretize_count % self.discretize_factor == 0.0:
            self.update_cpg_so2(phi)
        self.discretize_count += 1

    def generate_cpg_finite_size(self, cpg_length = 100000):
        # increase cpg length size and decrease 
        self.out0       = np.empty((1, cpg_length))
        self.out1       = np.empty((1, cpg_length))
        self.outFreq    = np.empty((1, cpg_length))

        for idx in range(cpg_length):
            self.update_cpg_with_discretize_factor(0.001, 1)       # not consider the perturbation
            self.out0[0][idx]    = self.get_out0() 
            self.out1[0][idx]    = self.get_out1()
        return {'out0':self.out0[0],
                'out1':self.out1[0]}

    def generate_cpg_one_cycle(self):
        self.generate_cpg_finite_size()
        cpg_cycle_index = self.zero_crossing_one_period(self.out0[0])

        out0_cpg_one_cycle = self.out0[0][cpg_cycle_index[0]:cpg_cycle_index[1]]
        out1_cpg_one_cycle = self.out1[0][cpg_cycle_index[0]:cpg_cycle_index[1]]

        return {'out0_cpg_one_cycle': out0_cpg_one_cycle,
                'out1_cpg_one_cycle': out1_cpg_one_cycle}

    def zero_crossing_one_period(self, signal):
        """
        Extract specific cycles (from start_cycle to end_cycle, inclusive) of a signal.

        Parameters:
            signal (numpy array): The input signal array.
            start_cycle (int): The first cycle to extract.
            end_cycle (int): The last cycle to extract.

        Returns:
            numpy array: The portion of the signal corresponding to the specified cycles,
                         or None if the cycles don't exist.
        """
        start_cycle = 3 
        end_cycle = 4
        
        # Identify the sign of the signal
        sign_signal = np.sign(signal)

        # Find zero-crossing indices
        crossings = np.where(np.diff(sign_signal) != 0)[0]

        # Check if enough crossings exist
        if len(crossings) < end_cycle:
            return None  # Not enough zero crossings to extract the requested cycles

        # Extract the indices for the requested cycles
        start_index = crossings[start_cycle - 1]
        end_index = crossings[end_cycle]

        # Return the signal corresponding to the specified cycles
        return [start_index, end_index + 1]

    
    def get_out0(self):
        return self.out0_t
    
    def get_out1(self):
        return self.out1_t
    
    def get_out2(self):
        return self.out2_t
    
    def get_fcpg(self):
        return self.phi/(2*np.pi)  # return fcpg
    
    def get_w2p(self):
        return self.w2p_t
    
    def get_w20(self):
        return self.w20_t
    
    def get_w02(self):
        return self.w02_t
    
    def get_w00(self):
        return self.w00
    
    def get_w01(self):
        return self.w01
    
    def get_w10(self):
        return self.w10
    
    def get_w11(self):
        return self.w11
    
    def print_all_param(self):
        print("out0_t: {}".format(self.out0_t))
        print("out1_t: {}".format(self.out1_t))
        print("out2_t: {}".format(self.out2_t))
        print("freq: {}".format(self.phi/(2*3.14)))
        print("alpha: {}".format(self.alpha))
        print("w20_t: {}".format(self.w20_t))
        print("w02_t: {}".format(self.w02_t))
        print("w2p_t: {}".format(self.w2p_t))
        print("learning rate: {}".format(self.hebbian_learning_rate))
        print("A02: {}".format(self.A02))
        print("A20: {}".format(self.A20))
        print("A2p: {}".format(self.A2p))
        print("B02: {}".format(self.B02))
        print("B20: {}".format(self.B20))
        print("B2p: {}".format(self.B2p))


