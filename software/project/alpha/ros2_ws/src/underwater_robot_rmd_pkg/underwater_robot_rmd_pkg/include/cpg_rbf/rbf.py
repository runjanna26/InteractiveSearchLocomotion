import numpy as np

'''
Parameters:
    nc (int)                    : Number of centers. Default is 100.
    variance_gaussian (float)   : Variance of the Gaussian distribution. Default is 0.01.
    nM (int)                    : Number of samples in the target trajectory. Default is 500.
    alpha (float)               : Learning rate. Default is 0.25.
'''
class RBF:
    def __init__(self, nc = 50, variance_gaussian = 0.01, nM = 500, alpha = 0.25): 
        self.nc = nc
        self.variance_gaussian = variance_gaussian
        self.nM = nM
        self.alpha = alpha

        self.target_length = None
        self.ci = []
        self.cx = []
        self.cy = []

        self.M  = []
        self.M_stack = []

        self.error = []
        self.error_new_traj = []

        self.error_stack = []
        self.error_max = 0
        self.error_max_stack = []

        self.W_stack = []

        self.learning_iteration = 0
        self.learning_rate = 0.25
        self.learning_rate_2 = 0.5

    ######################################################################## 
    #                        Record the trajectory
    # [1] Construct the kernels with CPG one cycle
    # [2] Imitate the path by learning
    ######################################################################## 
    def construct_kernels_with_cpg_one_cycle(self, O0_cpg_one_cycle, O1_cpg_one_cycle, target_length):
        '''
        Parameters:
            O0_cpg_one_cycle : output 0 of CPG one cycle
            O1_cpg_one_cycle : output 1 of CPG one cycle
            target_length : target length of the trajectory
        '''
        if target_length is None:
            print('please input target length')
            return 

        self.target_length = target_length

        self.K = np.zeros((self.nc, self.target_length))
        self.W = np.zeros((self.nc))
        self.M = np.zeros((self.target_length))

        # Centerize the kernel by using a CPG one cycle
        self.ci = np.linspace(0, O0_cpg_one_cycle.shape[0]-1, num = self.nc, dtype = int) # self.ci is the indices of raw data center at each sampling point
        self.cx = O0_cpg_one_cycle[self.ci]
        self.cy = O1_cpg_one_cycle[self.ci]


        b = np.zeros((self.target_length, 1))
        for i in range(self.nc):
            b = np.exp(-(np.power((O0_cpg_one_cycle - self.cx[i]), 2) + np.power((O1_cpg_one_cycle - self.cy[i]), 2)) / self.variance_gaussian) # b is a normalized gaussian distribution
            self.K[i, :] = b.transpose()
        return self.K
    
    def imitate_path_by_learning(self, target_traj, max_learning_iteration = 500, learning_rate = 0.05):
        '''
        Parameters:
            target_traj : target trajectory (length 10000 samples)
            max_learning_iteration : maximum learning iteration. Default is 500.
            learning_rate : learning rate. Default is 0.05.
        '''
        
        ''' learning rate 0.25 is too high '''

        self.learning_iteration = max_learning_iteration
        self.learning_rate = learning_rate

        # self.M_stack = [] # Comment to see the evolution across the learning serveral paths
        # self.error_stack, error_max_stack = [] # Comment to see the evolution across the learning serveral paths
        for i in range(self.learning_iteration):
            self.M = np.matmul(self.W, self.K)           
            
            self.error = (target_traj[self.ci] - self.M[self.ci])
            
            self.W = self.W + learning_rate * self.error   # [where is the point of has much error --> adjust weight]

            # Investigate path learning evolution
            self.M_stack.append(self.M)
            self.W_stack.append(self.W)


            self.error_max = np.mean(abs(self.error))  # Get the maximum absolute error --> [should change to average error along path]  --> error 10% of first iteration is defined as converge.
            self.error_max_stack.append(self.error_max)
            self.error_stack.append(self.error)
            if self.error_max < 0.01:
                print(f'last iteration: {i}')
                break
        return self.M  
    
    def adapt_imitate_path(self, qd_old, q_old, max_learning_iteration = 10, learning_rate = 0.05):
        '''
        qd_old :    old desired (traj.) gait cycle (length 10000 samples)
        q_old  :    old actual (traj.) gait cycle (length 10000 samples)
        '''

        self.learning_iteration = max_learning_iteration
        self.learning_rate_2 = learning_rate

        # if qd_old != self.target_length and q_old != self.target_length:
        #     print('length is not match, please check the length of traj.')
        #     return
        
    
        # # Solution 1
        # for i in range(self.learning_iteration):
        #     self.M = np.matmul(self.W, self.K)           
        #     self.error = (qd_old[self.ci] - self.M[self.ci] + q_old[self.ci] )
        #     self.W = self.W + learning_rate * self.error  
            
        # Solution 2
        self.error_new_traj = qd_old[self.ci] - q_old[self.ci]
        self.W = self.W + self.learning_rate_2 * self.error_new_traj  
        self.M = np.matmul(self.W, self.K)
            
        return self.M 


    def calualate_weight_mul_kernel(self):
        return np.matmul(self.W, self.K)
    def update_rbf_weigt(self, new_weight):
        self.W = new_weight
    def get_imitated_path(self):
        return self.M


    def get_rbf_weight(self):
        return self.W
    def get_evolution_learning_path(self):
        return np.asarray(self.M_stack)
    def get_error_while_learning(self):
        return np.asarray(self.error_stack)
    def get_error_max_while_learning(self):
        return np.asarray(self.error_max_stack)
    def get_weight_while_learning(self):
        return np.asarray(self.W_stack)

    
    ######################################################################## 
    #                        Replay the trajectory
    # [1] Regenerate the target trajectory
    ######################################################################## 
    def regenerate_target_traj(self, O0_cpg_t, O1_cpg_t, weight):  
        '''
        Reconstruct kernels with CPG on time step using vectorized operations
        '''

        self.K = np.zeros((self.nc,1))
        for i in range(self.nc):
            b = np.exp(-(np.power((O0_cpg_t - self.cx[i]), 2) + np.power((O1_cpg_t - self.cy[i]), 2)) / self.variance_gaussian) # b is a normalized gaussian distribution
            self.K[i] = b

        self.M = np.matmul(weight, self.K)     
        return self.M[0]    
