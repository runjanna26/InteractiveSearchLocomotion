import numpy as np
from cpg_afdc import CPG_AFDC
from rbf import RBF 

from scipy.interpolate import interp1d



class CPG_RBF:
    def __init__(self):
        self.cpg = CPG_AFDC()
        self.rbf = RBF()

        self.cpg_one_cycle = np.load('cpg_one_cycle.npz')
        
        self.target_weight = None

    def resample_cycle(self, cycle, num_points):
        """Resamples a gait cycle to have a fixed number of points using interpolation."""
        original_indices = np.linspace(0, 1, len(cycle))  # original indices based on the length of the cycle
        new_indices = np.linspace(0, 1, num_points)  # new indices to match the desired number of points
        interpolation_function = interp1d(original_indices, cycle, kind='linear')  # interpolate the data
        return interpolation_function(new_indices)  # return the resampled cycle
    
    def offline_imitate_path(self, path):

        target = self.resample_cycle(np.array(path), 10000)     
        target_length = len(target)

        cpg_one_cycle = self.cpg.generate_cpg_one_cycle()

        out0_cpg_one_cycle = self.resample_cycle(cpg_one_cycle['out0_cpg_one_cycle'], target_length)    
        out1_cpg_one_cycle = self.resample_cycle(cpg_one_cycle['out1_cpg_one_cycle'], target_length)  

        cpg_kernel = self.rbf.construct_kernels_with_cpg_one_cycle(out0_cpg_one_cycle, out1_cpg_one_cycle, target_length)
        imitated_path = self.rbf.imitate_path_by_learning(target)

        self.target_weight = self.rbf.get_rbf_weight()


        return {'imitated_path' : imitated_path,
                'target_weight' : self.target_weight,
                'cpg_kernel'    : cpg_kernel}
    
    def online_imitate_path(self, path):
        target = self.resample_cycle(np.array(path), 10000)     
        target_length = len(target)
        
        
        out0_cpg_one_cycle = self.resample_cycle(self.cpg_one_cycle['O0'], target_length)    
        out1_cpg_one_cycle = self.resample_cycle(self.cpg_one_cycle['O1'], target_length)  

        cpg_kernel = self.rbf.construct_kernels_with_cpg_one_cycle(out0_cpg_one_cycle, out1_cpg_one_cycle, target_length)
        imitated_path = self.rbf.imitate_path_by_learning(target)

        self.target_weight = self.rbf.get_rbf_weight()


        return {'imitated_path' : imitated_path,
                'target_weight' : self.target_weight,
                'cpg_kernel'    : cpg_kernel}
        
    
    def online_regenerate_path(self, cpg_o0_t, cpg_o1_t):
        if self.target_weight is None:
            return
        self.rbf.regenerate_target_traj(cpg_o0_t, cpg_o1_t, self.target_weight)
        