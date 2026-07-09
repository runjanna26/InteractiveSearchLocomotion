import numpy as np
import matplotlib.pyplot as plt

'''

    []

'''
class CPG_AFDC:
    def __init__(self,
                 o0_init    = 0.2, 
                 o1_init    = 0.2, 
                 o2_init    = 0.2, 
                 phi_init   = 0.01 * 2 * np.pi, 
                 _alpha     = 1.01, 
                 lrate      = 1.3):  # 1.3
        
        self.out0_t = o0_init
        self.out1_t = o1_init
        self.out2_t = o2_init
        self.phi    = phi_init

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
        
        self.B02 = 0.01 * factor  # High is stable but slower
        self.B20 = 0.01 * factor
        self.B2p = 0.01 * factor
        

        
        self.w00 = self.alpha*np.cos(self.phi)
        self.w01 = self.alpha*np.sin(self.phi)
        self.w10 = self.alpha*(-np.sin(self.phi))
        self.w11 = self.alpha*np.cos(self.phi)
        

        self.discretize_count = 0
        self.discretize_factor = 10



        self.w20_t1 = 0.0
        self.w02_t1 = 0.0
        self.s = 1.0


        self.out0_t1 = 0.0
        self.out1_t1 = 0.0
        self.out2_t1 = 0.0



    # CPG-AFDC
    def update_adaptive_cpg_with_synaptic_plasticity(self, perturbation: float):
        # Adaptive CPG with Synaptic Plasticity (H0, H1, H2) in eq.(2)
        self.out0_t1 = np.tanh(self.w00*self.out0_t + self.w01*self.out1_t + self.w02_t*self.out2_t)
        self.out1_t1 = np.tanh(self.w10*self.out0_t + self.w11*self.out1_t)
        self.out2_t1 = np.tanh(self.w20_t*self.out0_t + self.w2p_t*(perturbation))  

        # update cpg phi with "Adaptation throgh Fast Dynamical Coupling (AFDC)" in eq.(4)
        self.delta_phi = self.hebbian_learning_rate*(self.w02_t*self.out2_t*self.w01*self.out1_t)       # Hebbian learning rule
        self.phi = self.phi + self.delta_phi                                                            # Stocastic Gradient Descent (new = old + (lrate*error))

        # update cpg weight 
        self.w00 = self.alpha * np.cos(self.phi)
        self.w01 = self.alpha * np.sin(self.phi) 
        self.w10 = self.alpha * (-np.sin(self.phi)) 
        self.w11 = self.alpha * np.cos(self.phi)  


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


        # save for next iteration
        self.out0_t = self.out0_t1
        self.out1_t = self.out1_t1
        self.out2_t = self.out2_t1  
        self.w20_t = self.w20_t1
        self.w02_t = self.w02_t1
        self.w2p_t = self.w2p_t1

        return self.phi
    
    def update_adaptive_cpg_with_synaptic_plasticity_list(self, perturbation_cycle: np.ndarray, duplicated_num_cycles = 20, target_ptp = 0.18):
        phi_list = []
        cycle_list = []
        
        # Scale the perturbation cycle to have a specific peak-to-peak amplitude
        current_ptp = np.ptp(perturbation_cycle)
        if current_ptp == 0:
            raise ValueError("Cannot scale: peak-to-peak amplitude is zero (flat signal).")
        scale_factor = target_ptp / current_ptp

        perturbation_cycle = perturbation_cycle*scale_factor


        # Simulate multiple cycles by duplicating a input cycle
        perturbation_cycles = np.tile(perturbation_cycle, (duplicated_num_cycles, 1)) if perturbation_cycle.ndim > 1 else np.tile(perturbation_cycle, duplicated_num_cycles)
        


        for perturbation in perturbation_cycles:
            phi = self.update_adaptive_cpg_with_synaptic_plasticity(perturbation)
            phi_list.append(phi)
            cycle_list.append(perturbation)

        

        # phi_array = np.array(phi_list)
        # cycle_array = np.array(cycle_list)

        # # ---- PLOT SECTION ----
        # plt.figure(figsize=(8, 4))
        # plt.plot(phi_array, marker='o')
        # plt.plot(cycle_array, alpha=0.3)
        # plt.hlines(0.08, xmin=0, xmax=len(phi_array)-1, colors='r', linestyles='dashed', label='Zero Line')
        # plt.title("Evolution of $\phi$ over Gait Cycles")
        # plt.xlabel("Cycle index")
        # plt.ylabel("$\phi$")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()
        # # ----------------------
        return np.mean(phi_list)
        


    # ========================== API ==========================
    # def update_cpg_with_discretize_factor(self, set_fcpg, discretize_factor):

    #     # freq.[Hz] = f_cpg[cpg_cycles per iteration] * 1/discretize_factor[iterations per program step] * update_rate[program step per second]

    #     self.discretize_factor = discretize_factor
    #     phi = set_fcpg * 2 * np.pi                                  
    #     if self.discretize_count % self.discretize_factor == 0.0:
    #         ...
    #         # self.update_cpg_so2(phi)
    #     self.discretize_count += 1

    # def generate_cpg_finite_size(self, phi, cpg_length = 100000):
    #     # increase cpg length size and decrease 
    #     self.out0       = np.empty((1, cpg_length))
    #     self.out1       = np.empty((1, cpg_length))
    #     self.outFreq    = np.empty((1, cpg_length))

    #     for idx in range(cpg_length):
    #         self.update_cpg_so2(phi)       # not consider the perturbation
    #         self.out0[0][idx]    = self.get_out0() 
    #         self.out1[0][idx]    = self.get_out1()
    #     return {'out0':self.out0[0],
    #             'out1':self.out1[0]}

    # def generate_cpg_one_cycle(self, phi):
    #     self.generate_cpg_finite_size(phi)
    #     cpg_cycle_index = self.zero_crossing_one_period(self.out0[0])

    #     out0_cpg_one_cycle = self.out0[0][cpg_cycle_index[0]:cpg_cycle_index[1]]
    #     out1_cpg_one_cycle = self.out1[0][cpg_cycle_index[0]:cpg_cycle_index[1]]

    #     return {'out0_cpg_one_cycle': out0_cpg_one_cycle,
    #             'out1_cpg_one_cycle': out1_cpg_one_cycle}

    def zero_crossing_one_period(self, signal , value = 0.0):
        """
        Extract specific cycles (from start_cycle to end_cycle, inclusive) of a signal,
        based on crossings at a specific value.

        Parameters:
            signal (numpy array): The input signal array.
            value (float): The value at which to detect crossings. Default is 0.
            
        Returns:
            list: The start and end indices corresponding to the specified cycles,
                or None if the cycles don't exist.
        """
        start_cycle = 3 
        end_cycle = 4
        
        # Identify if signal is above or below the crossing value
        sign_signal = np.sign(signal - value)

        # Find value-crossing indices
        crossings = np.where(np.diff(sign_signal) != 0.0)[0]

        # Check if enough crossings exist
        if len(crossings) < end_cycle:
            return None  # Not enough crossings to extract the requested cycles

        # Extract the indices for the requested cycles
        start_index = crossings[start_cycle - 1]
        end_index = crossings[end_cycle]

        # Return the signal corresponding to the specified cycles
        return [start_index, end_index + 1]



    # # Only CPG
    # def update_cpg_so2(self, phi):
    #     self.phi = phi
        
    #     self.out0_t1 = self.s * np.tanh(self.w00*self.out0_t + self.w01*self.out1_t)
    #     self.out1_t1 = self.s * np.tanh(self.w10*self.out0_t + self.w11*self.out1_t)

    #     # self.w00 = self.alpha * np.cos(np.pi/2)
    #     # self.w01 = self.alpha * np.sin(np.pi/2) 
    #     # self.w10 = self.alpha * (-np.sin(np.pi/2)) 
    #     # self.w11 = self.alpha * np.cos(np.pi/2)  

    #     # self.w00 = self.alpha * np.cos(np.pi/2)
    #     # self.w01 = self.alpha * np.sin(0) 
    #     # self.w10 = self.alpha * (-np.sin(0)) 
    #     # self.w11 = self.alpha * np.cos(np.pi/2) 

    #     # # update cpg weight 
    #     # self.w00 = self.alpha * np.cos(1.47)
    #     # self.w01 = self.alpha * np.sin(0) 
    #     # self.w10 = self.alpha * (-np.sin(0)) 
    #     # self.w11 = self.alpha * np.cos(1.47)  

    #     # save for next iteration
    #     self.out0_t = self.out0_t1
    #     self.out1_t = self.out1_t1
    #     self.w20_t = self.w20_t1
    #     self.w02_t = self.w02_t1
        