import numpy as np

#DIL Class
class DIL:
    def __init__(self, Af, Bf, Cf, As, Bs, Cs):
        '''
        Parameters:
            Af (float) : low-pass filter coefficient for fast learning
            As (float) : low-pass filter coefficient for slow learning
            
            Bf (float) : gain coefficient for fast learning
            Bs (float) : gain coefficient for slow learning
            
            Cf (float) : integral gain coefficient for fast learning
            Cs (float) : integral gain coefficient for slow learning
        '''
        self.Af = Af
        self.Bf = Bf
        self.Cf = Cf
        
        self.As = As
        self.Bs = Bs
        self.Cs = Cs
        
        self.sum_err = 0.0    # sum of average error
        self.xf_prev = 0.0    # previous value of xf
        self.xs_prev = 0.0    # previous value of xs

    def calculate_DIL(self, avg_err):
        self.sum_err = self.sum_err + avg_err
        xf = self.Af * self.xf_prev+ self.Bf*avg_err + self.Cf * self.sum_err
        xs = self.As * self.xs_prev+ self.Bs*avg_err + self.Cs * self.sum_err
        output = xf+ xs
        self.xf_prev = xf
        self.xs_prev = xs
        # print("xf, xs:", xf,xs)
        return  output, xf, xs