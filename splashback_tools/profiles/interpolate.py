import numpy as np
class Interpolate:
    def __init__(self, xs, **ys):

        self.xs = xs

        if not isinstance(xs, np.ndarray):
            raise Exception("x not an array")
        
        self.n_samp = len(xs)

        y_vals = ys.values()
        if not all(isinstance(y_val, np.ndarray) for y_val in y_vals):
            raise Exception("Ys not arrays")
        if not all(len(y_val) == self.n_samp for y_val in y_vals):
            raise Exception("Ys not same length")

        self.ys = ys

    def interpolate(self, x, paramname):
        
        #Can't interpolate if too small
        if x < self.xs[0] or x > self.xs[-1]:
            return float("nan")
    
        #Use binary search to find interval containing x
        l = 0
        r = len(self.xs) - 1
        flrInd = None

        while (l < r):
            flrInd = (l + r)//2
            if self.xs[flrInd] <= x and self.xs[flrInd] >= x:
                break
            if self.xs[flrInd] > x:
                r = flrInd - 1
            else:
                l = flrInd + 1

        #Linear interpolate between the two points
        x1 = self.xs[flrInd]
        x2 = self.xs[flrInd + 1]
        y1 = self.ys[paramname][flrInd]
        y2 = self.ys[paramname][flrInd + 1]

        return (x - x1) * (y2 - y1)/(x2 - x1) + y1

