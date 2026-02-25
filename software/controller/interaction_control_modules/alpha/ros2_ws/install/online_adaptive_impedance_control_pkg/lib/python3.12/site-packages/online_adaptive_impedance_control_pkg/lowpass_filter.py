class LPF:
    def __init__(self, alpha):
        self.alpha = alpha
        self.prev_y = 0.0
        self.prev_x = 0.0
        
    def filter(self, x):
        y = (1 - self.alpha) * self.prev_y + self.alpha * (x + self.prev_x) / 2
        self.prev_y = y
        self.prev_x = x
        return y
    
    def reset(self):
        self.prev_y = 0.0
        self.prev_x = 0.0

