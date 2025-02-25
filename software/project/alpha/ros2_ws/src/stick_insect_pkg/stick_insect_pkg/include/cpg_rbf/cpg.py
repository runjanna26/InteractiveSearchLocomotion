import numpy as np


class CPG:
    def __init__(self, MI=None, alpha=None):
        # Setting up default values
        self.b1 = self.b2 = 0.01  # Bias term
        self.g1 = self.g2 = 0.0   # Feedback strength
        self.reflex = 0.18        # Reflex
        self.phaseAdaptationActive = False
        self.output = np.zeros(2)
        self.a1 = self.a2 = 0.0
        self.w11 = self.w22 = 1.0
        
        # Handling different constructors
        if MI is not None and alpha is not None:
            self.g1 = self.g2 = alpha
            self.reflex = MI
        elif MI is not None:
            self.g1 = self.g2 = 0.1
            self.reflex = MI

    def Tanh(self, input):
        return (2.0 / (1.0 + np.exp(-2.0 * input))) - 1.0

    def setReflex(self, input):
        self.reflex = input

    def getCPGValues(self):
        return self.output

    def getActivation(self):
        return np.array([self.a1, self.a2])

    def enableForceFeedback(self):
        self.phaseAdaptationActive = True

    def disableForceFeedback(self):
        self.phaseAdaptationActive = False

    def calculate(self):
        self.w12 = 0.18 + self.reflex
        self.w21 = -0.18 - self.reflex

        self.a1 = self.w11 * self.output[0] + self.w12 * self.output[1] + self.b1
        self.a2 = self.w22 * self.output[1] + self.w21 * self.output[0] + self.b2

        self.output[0] = self.Tanh(self.a1)
        self.output[1] = self.Tanh(self.a2)

        return self.output

    def calculatePhaseAdaptation(self, fs):
        self.w12 = 0.18 + self.reflex
        self.w21 = -0.18 - self.reflex

        force = 0
        if self.phaseAdaptationActive:
            force = fs  # Or use: force = np.minimum(fs, 1.0)

        self.a1 = (self.w11 * self.output[0] + self.w12 * self.output[1] + self.b1
                   - self.g1 * force * np.cos(self.a1))
        self.a2 = (self.w22 * self.output[1] + self.w21 * self.output[0] + self.b2
                   - self.g2 * force * np.sin(self.a2))

        self.output[0] = self.Tanh(self.a1)
        self.output[1] = self.Tanh(self.a2)

        return self.output
