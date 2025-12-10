import math

class LowPassFilter:
    def __init__(self, alpha, initval=None):
        self.y = initval
        self.s = False if initval is None else True
        self.alpha = alpha
    def filter(self, x):
        if not self.s:
            self.y = x
            self.s = True
            return x
        self.y = self.alpha * x + (1.0 - self.alpha) * self.y
        return self.y

def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)

def exponential_smoothing(a, x, prev):
    return a * x + (1 - a) * prev

class OneEuro:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff  # minimum cutoff frequency
        self.beta = beta              # speed coefficient
        self.d_cutoff = d_cutoff      # cutoff for derivative
        self.x_prev = None
        self.t_prev = None
        self.dx_filter = None
        self.x_filter = None

    def update(self, x, t):
        if self.t_prev is None or self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            self.dx_filter = LowPassFilter(1.0, 0.0)
            self.x_filter = LowPassFilter(1.0, x)
            return x

        # Compute time elapsed
        t_e = t - self.t_prev
        self.t_prev = t

        # Compute derivative of signal
        dx = (x - self.x_prev) / t_e if t_e > 0 else 0.0
        self.x_prev = x

        # Filter derivative
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx_hat = self.dx_filter.filter(dx)

        # Adjust cutoff frequency based on speed
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Filter signal
        a = smoothing_factor(t_e, cutoff)
        x_hat = self.x_filter.filter(x)

        return x_hat
