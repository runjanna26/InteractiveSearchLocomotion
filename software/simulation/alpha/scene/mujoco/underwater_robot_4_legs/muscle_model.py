import numpy as np

# ==========================================
# 1. ADAPTIVE CONTROLLER CLASS
# ==========================================
class MuscleModel:
    def __init__(self, _a, _b, _beta, _init_pos):
        self.pos_init = _init_pos
        self.a = _a       # Adaptation scalar (Base stiffness)
        self.b = _b       # Sensitivity to error
        self.beta = _beta # Weight of velocity error

        self.pos_des = 0.0
        self.pos_fb = 0.0
        self.pos_des_prev = 0.0
        self.vel_des = 0.0
        self.vel_fb = 0.0
        
        self.safe_dt = 1e-3
        
        self.K, self.D, self.F = 0.0, 0.0, 0.0
        
        self.alpha_vel = 0.1
        self.DE = 0.0

    def calculate(self, pos_des, pos_fb, vel_fb, dt):
        self.pos_des = pos_des
        self.pos_fb = pos_fb
        self.vel_fb = vel_fb
        
        # Derivative of desired position
        self.safe_dt = dt 
        if dt < 1e-6:
            self.safe_dt = 1e-3 
            print(f"Warning: dt too small ({dt:.6f}s). Using safe_dt={self.safe_dt:.3f}s for velocity calculation.")
        
        vel_des_raw = (self.pos_des - self.pos_des_prev) / self.safe_dt
        self.vel_des = (self.alpha_vel * vel_des_raw) + ((1.0 - self.alpha_vel) * self.vel_des)
        # Adaptive Scalar Law
        adapt_scalar = self.gen_adapt_scalar()
        if abs(adapt_scalar) < 1e-6: adapt_scalar = 1e-6

        # Calculate Gains (F, K, D)
        self.F = self.gen_track_error() / adapt_scalar
        self.K = self.F * self.gen_pos_error()
        self.D = self.F * self.gen_vel_error()

        # self.K = 35
        # self.D = 0.5

        self.pos_des_prev = self.pos_des

    def gen_pos_error(self): return (self.pos_fb - self.pos_init) - self.pos_des
    def gen_vel_error(self): return self.vel_fb - self.vel_des
    def gen_track_error(self): return self.gen_pos_error() + self.beta * self.gen_vel_error()
    def gen_adapt_scalar(self): 
        return self.a / (1.0 + (self.b * (np.abs(self.gen_track_error()))**2))
    
    def get_torque(self):
        # Limits to prevent explosion
        # self.K = np.clip(self.K, 0.0, 500.0)
        # self.D = np.clip(self.D, 0.0, 5.0)
        # self.F = np.clip(self.F, -20.0, 20.0)

        # Torque Command
        return -self.F - (self.K * self.gen_pos_error()) - (self.D * self.gen_vel_error())
    
    def get_power_damping(self):
        # Energy dissipated by damping: E_damp = ∫ D(t) * (q(t))^2 dt
        # self.DE += self.D * (self.vel_fb ** 2) * self.safe_dt
        
        self.DE = self.D * (self.vel_fb ** 2) # power damping at current time step
        return self.DE
