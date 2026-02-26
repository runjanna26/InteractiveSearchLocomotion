import numpy as np

class CPG_SO2:
    def __init__(self,
                 o0_init    = 0.14, 
                 o1_init    = 0.14, 
                 ϕ_init     = 0.01 * 2 * np.pi, 
                 _alpha     = 1.01):
        '''
        Parameters:
            o0_init (float): Initial output value for neuron 0. Default is 0.2.
            o1_init (float): Initial output value for neuron 1. Default is 0.2.
            ϕ_init (float): Initial phase difference between neurons in radians. Default is 0.01 * 2 * np.pi.
            _alpha (float): Scaling factor for feedback. Default is 1.01.   
        '''
        # Initial output values for the CPG neurons
        self.out0_t = o0_init
        self.out1_t = o1_init
        
        # Initial ϕ difference between neurons
        self.ϕ    = ϕ_init
        
        # Scaling factor for feedback
        self.alpha = _alpha
        
        # Initialize connection weights between neurons
        self.w00 = self.alpha * np.cos(self.ϕ)
        self.w01 = self.alpha * np.sin(self.ϕ)
        self.w10 = self.alpha * (-np.sin(self.ϕ))
        self.w11 = self.alpha * np.cos(self.ϕ)
        
        self.a0_t = 0.0
        self.a1_t = 0.0
     
        self.a0_t1 = 0.0
        self.a1_t1 = 0.0   
        # Initialize output weights for the CPG neurons
        self.output_cpg_weight = 1.0
        
        self.s0 = 0.0
        self.s1 = 0.0
    
    def update_cpg(self, ϕ):
        '''
        Parameters:
            ϕ (float): Frequency of CPG outputs.
                if ϕ is None : you should set the weights by yourself.
        '''
        self.ϕ = ϕ
        if self.ϕ is not None:
            # Update the weights for the next cycle        
            self.w00 = self.alpha *   np.cos(self.ϕ)
            self.w01 = self.alpha *   np.sin(self.ϕ) 
            self.w10 = self.alpha * (-np.sin(self.ϕ)) 
            self.w11 = self.alpha *   np.cos(self.ϕ) 
        
        # Calculate the new output for neuron 0 and neuron 1
        self.a0_t1    = self.w00*self.out0_t + self.w01*self.out1_t - self.s0 * np.cos(self.a0_t)
        self.a1_t1    = self.w10*self.out0_t + self.w11*self.out1_t - self.s1 * np.sin(self.a1_t)
        self.out0_t1 = self.output_cpg_weight * np.tanh(self.a0_t1)
        self.out1_t1 = self.output_cpg_weight * np.tanh(self.a1_t1)

 
        # Save outputs for the next iteration
        self.out0_t = self.out0_t1
        self.out1_t = self.out1_t1
        self.a0_t = self.a0_t1
        self.a1_t = self.a1_t1
        
    ######################################################################## 
    #               API for generating CPG one cycle
    ######################################################################## 
    
    def generate_cpg_one_cycle(self, ϕ):
        '''
        Parameters:
            ϕ (float): Frequency of CPG outputs.
        '''
        # Generate CPG outputs for a finite size
        self.generate_cpg_finite_size(ϕ)
            
        # Find the indices for one cycle using zero-crossing method
        cpg_cycle_index = self.zero_crossing_one_period(self.out0[0])

        # Extract one cycle of the CPG outputs for both neurons
        out0_cpg_one_cycle = self.out0[0][cpg_cycle_index[0]:cpg_cycle_index[1]]
        out1_cpg_one_cycle = self.out1[0][cpg_cycle_index[0]:cpg_cycle_index[1]]
        

        # Return the extracted cycle for further analysis or plotting
        return {'out0_cpg_one_cycle': out0_cpg_one_cycle,
                'out1_cpg_one_cycle': out1_cpg_one_cycle}
        
    def generate_cpg_finite_size(self, ϕ, cpg_length = 100000):
        '''
        Parameters:
            ϕ (float): Frequency of CPG outputs.
            cpg_length (int): Length of the CPG sequence to generate. Default is 100000.
        '''
        # Prepare arrays to store outputs for the entire CPG sequence
        self.out0       = np.empty((1, cpg_length))
        self.out1       = np.empty((1, cpg_length))
        
        # Generate CPG outputs over the specified length
        for idx in range(cpg_length):
            self.update_cpg(ϕ)       # Update CPG without perturbation
            self.out0[0][idx]    = self.out0_t  # Store output for neuron 0
            self.out1[0][idx]    = self.out1_t  # Store output for neuron 1
        # Return the generated sequences for both neurons
        return {'out0':self.out0[0],
                'out1':self.out1[0]}
    
    def zero_crossing_one_period(self, signal, value=0.0):
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
        # Define which cycles to extract (3rd cycle to 4th cycle in this case)
        start_cycle = 3 
        end_cycle = 4
        
        # Determine if each point in the signal is above or below the specified value
        # This converts the signal to -1 (below) or +1 (above) relative to the crossing value
        sign_signal = np.sign(signal - value)

        # Identify indices where the signal crosses the specified value
        # This happens when the sign changes from positive to negative or vice versa
        crossings = np.where(np.diff(sign_signal) != 0.0)[0]

        # Check if enough crossings exist to extract the required cycles
        if len(crossings) < end_cycle:
            # If not enough crossings, return None indicating the cycles can't be extracted
            return None  

        # Extract indices corresponding to the start and end of the requested cycles
        # `start_index` is the point just before the signal crosses at the start of the cycle
        start_index = crossings[start_cycle - 1]
        # `end_index` is the point just before the signal crosses at the end of the cycle
        end_index = crossings[end_cycle]

        # Return the indices of the signal corresponding to the requested cycles
        # Adding 1 to `end_index` to include the last point in the cycle
        return [start_index, end_index + 1]
    
    
    
class CPG_LOCO:
    def __init__(self):
        self.cpg = CPG_SO2()
    
    def modulate_cpg(self, ϕ, α, β):
        '''
        Parameters:
            ϕ  (float): Frequency of CPG outputs.
            α  (float): Pause input for the CPG {0,1}. Default is 0.0.
            β  (float): Rewind input for the CPG {-1,1}. Default is 0.0.
        '''
        
        # Adaptive Weights
        ϕ00 = (α * np.pi/2.01) + ((1 - α) * β *  ϕ)
        ϕ01 = (1 - α) * β *  ϕ
        ϕ10 = (1 - α) * β *  ϕ
        ϕ11 = (α * np.pi/2.01) + ((1 - α) * β *  ϕ)
        
        self.cpg.w00 = self.cpg.alpha *  np.cos(ϕ00)
        self.cpg.w01 = self.cpg.alpha *  np.sin(ϕ01)
        self.cpg.w10 = self.cpg.alpha * -np.sin(ϕ10)
        self.cpg.w11 = self.cpg.alpha *  np.cos(ϕ11)
        
        self.cpg.output_cpg_weight = (α  / (self.cpg.alpha * np.cos(np.pi/2.01))) + (1 - α) * (np.abs(β))
        

        # Update CPG outputs
        self.cpg.update_cpg(None)
            
        return {'cpg_output_0': self.cpg.out0_t,
                'cpg_output_1': self.cpg.out1_t}
