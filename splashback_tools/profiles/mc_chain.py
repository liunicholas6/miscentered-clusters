import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class McChain:
    def __init__(self, weights, loglikes, **samples):
        sample_vals = samples.values()
        
        if not all(isinstance(sample_val, np.ndarray) for sample_val in sample_vals):
            raise Exception("Samples not arrays")
        
        self.n_samp = len(next(iter(sample_vals)))

        if not all(len(sample_val) == self.n_samp for sample_val in sample_vals):
            raise Exception("Samples not same length")
        
        if weights is None:
            weights = np.ones(self.n_samp)
        if len(weights) != self.n_samp:
            raise Exception("Weights wrong length")
        if loglikes is None:
            loglikes = np.ones(self.n_samp)
        if len(loglikes) != self.n_samp:
            raise Exception("Loglikes wrong length")
        self.weights = weights
        self.loglikes = loglikes
        self.samples = samples
        self.sample_stats = {}

    def get_samp_stats(self, paramname):
        if not paramname in self.samples.keys():
            raise Exception(f"No parameter with name {paramname}")
        values = self.samples[paramname]

        #Lazy evaluate
        if paramname not in self.sample_stats:
            mean = np.average(values, weights=self.weights, axis = 0)
            var = np.average((values-mean)**2, weights=self.weights, axis = 0)
            std = np.sqrt(var)
            self.sample_stats[paramname] = (mean, std)
        
        return self.sample_stats[paramname]

    def add_samps(self, **samps):
        for k, v in samps.items():
            if len(v) != self.n_samp:
                raise Exception("Sample wrong length")
            self.samples[k] = v

    def plot(self, paramname, label = None, bounds = None, ax = plt.gca()):
        mean, std = self.get_samp_stats(paramname)

        #Plot 3 sigma if no  bounds specified
        if bounds is None:
            bounds = (mean - 3 * std, mean + 3 * std)
        
        minx, maxx = bounds
        x = np.linspace(minx, maxx, 500)
        if label is None:
            ax.plot(x, sp.stats.norm.pdf(x, mean, std) * std)
        else:
            ax.plot(x, sp.stats.norm.pdf(x, mean, std) * std, label = label)
        ax.yaxis.set_visible(False)
        ax.legend()

    def confidence_interval(self, paramname, stds = 1):
        mean, std = self.get_samp_stats(paramname)
        return (mean - std * stds, mean + std * stds)