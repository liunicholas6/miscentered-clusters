import numpy as np
import scipy as sp
from matplotlib.pyplot import Axes
import warnings

class McChain:
    def __init__(self, weights = None, loglikes = None, **samples):
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
        weights = self.weights

        if np.any(np.isnan(values)):
            warnings.warn(f"Samples of {paramname} contains NaN values")
            if paramname not in self.sample_stats:
                filt = ~np.isnan(values)
                values = values[filt]
                weights = weights[filt]

        #Lazy evaluate
        if paramname not in self.sample_stats:
            mean = np.average(values, weights=weights, axis = 0)
            var = np.average((values-mean)**2, weights=weights, axis = 0)
            std = np.sqrt(var)
            self.sample_stats[paramname] = (mean, std)
        
        return self.sample_stats[paramname]

    def add_samps(self, **samps):
        for k, v in samps.items():
            if len(v) != self.n_samp:
                raise Exception("Sample wrong length")
            self.samples[k] = v

    def plot_1d_posterior(self, paramname, ax : Axes, bounds = None, **kwargs):
        mean, std = self.get_samp_stats(paramname)

        #Plot 3 sigma if no bounds specified
        if bounds is None:
            bounds = (mean - 3 * std, mean + 3 * std)
        
        minx, maxx = bounds
        x = np.linspace(minx, maxx, 500)
        curve = ax.plot(x, sp.stats.norm.pdf(x, mean, std), **kwargs)[0]
        ax.yaxis.set_visible(False)
        ax.legend()

        #Plot histogram
        values = self.samples[paramname]
        filt = ~np.isnan(values)
        values = values[filt]
        weights = self.weights[filt]
        counts, bins = np.histogram(values, bins = 30, weights = weights, range = (minx - 2 * std, maxx + 2 * std))
        weights = counts/sum(counts)/(bins[1] - bins[0])
        ax.hist(bins[:-1], bins, weights=weights, color = curve.get_color(), alpha = 0.5)

    def plot_2d_scatter(self, p1, p2, ax : Axes, **kwargs):
        v1 = self.samples[p1]
        v2 = self.samples[p2]
        filt = ~(np.isnan(v1) | np.isnan(v2))
        v1 = v1[filt]
        v2 = v2[filt]
        weights = self.weights[filt]
        ax.scatter(v1, v2, alpha=weights, **kwargs)
        ax.legend()

    def confidence_interval(self, paramname, stds = 1):
        mean, std = self.get_samp_stats(paramname)
        return (mean - std * stds, mean + std * stds)