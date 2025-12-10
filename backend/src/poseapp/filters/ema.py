# src/poseapp/filters/ema.py
class EMA:
    def __init__(self, alpha=0.25):
        # Initialize smoothing factor (alpha ∈ [0,1]) and internal state
        self.alpha = alpha
        self.y = None  # previous filtered value

    def update(self, x):
        # Apply exponential moving average (EMA) filter
        if x is None: 
            return self.y  # if no input, return last known value
        # First value initializes y; then smooth using EMA formula
        self.y = x if self.y is None else (self.alpha * x + (1 - self.alpha) * self.y)
        return self.y  # return filtered result
