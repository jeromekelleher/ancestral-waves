"""
Module to run the simulations and collect results.
"""
from __future__ import print_function
from __future__ import division

import os
import sys
import glob
import math
import pickle
import random
import os.path
import tempfile
import argparse
import subprocess
import multiprocessing

import numpy as np

# Non essential packages
try:
    import fipy
    from scipy.integrate import ode
    from scipy.integrate import quad
    from scipy.interpolate import interp1d
    from scipy.special import lambertw
    from scipy.special import spence
    from scipy.optimize import curve_fit
    from scipy.optimize import brentq
    import matplotlib
    # Force matplotlib to not use any Xwindows backend.
    matplotlib.use('Agg')
    # Set the color cycle for plots using ColorBrewer
    # http://colorbrewer2.org/?type=qualitative&scheme=Set1&n=5
    matplotlib.rcParams['axes.color_cycle'] = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    #["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    #['r', 'k', 'c']
    from matplotlib import pyplot
except ImportError:
    print("Non simulation-essential package import failed")

import ercs
import discsim

MODEL_WRIGHT_FISHER = "Wright-Fisher"
MODEL_MORAN = "Moran"

# Used to control the amount of parallelism
__num_worker_processes = 1

def reset_cpu_affinity():
    """
    Numpy does some horrible things with CPU affinity. For some reason,
    numpy sets the CPU affinity to 0 on initialisation, meaning that
    _all_ subprocesses are pinned to CPU0. This makes using
    multiprocessing pointless. We use taskset to resolve this
    using taskset.
    """
    num_cpus = multiprocessing.cpu_count()
    cmd = ["taskset",  "-p", hex(2**num_cpus - 1), str(os.getpid())]
    with open(os.devnull, "w") as devnull:
        subprocess.check_call(cmd, stdout=devnull)

def dilogarithm(z):
    """
    Returns the dilogarithm function Li2(z)
    """
    Li2 = spence(1 - z)
    return Li2

def position_gauss(t, rho_sigma):
    """
    Returns the position of the Gaussian approximation at time t with
    parameters rho*sigma.
    """
    def f(x):
        t1 = -2.0**t * np.exp(-x**2 / (2 * t)) * x**2
        t2 = np.sqrt(2 * np.pi * t) * (x**2 - t) * rho_sigma
        return t1 + t2
    a = 0
    b = t * np.sqrt(2 * np.log(2))
    return brentq(f, a, b)

def gaussian_approximation(x, t, rho_sigma):
    """
    Returns the value of the Gaussian approximation at the specified time
    and location for the specified compound parameter rho*sigma
    """
    numer = 2.0**t * np.exp(-x**2 / (2 * t))
    denom = rho_sigma * np.sqrt(2 * np.pi * t)
    return 1 - np.exp(-numer / denom)


def rho_e_to_u(rho_e, r=1, nu=2, d=1):
    """
    Returns the value of u corresponding to the specified effective
    density.
    """
    assert d in [1, 2]
    Vd = 2 * r if d == 1 else math.pi * r**2
    return nu / (2 * rho_e * Vd)


def u_to_rho_e(u, r=1, nu=2, d=1):
    """
    Returns the value of the effective density for the specified value
    of u.
    """
    assert d in [1, 2]
    Vd = 2 * r if d == 1 else math.pi * r**2
    return nu / (2 * u * Vd)

def rho_e_to_deme_size(rho_e, model):
    """
    Returns the deme size corresponding to the specified discrete model.
    """
    ret = 2 * rho_e
    if model == MODEL_MORAN:
        ret = 4 * rho_e
    return ret

def deme_size_to_rho_e(N, model):
    """
    Returns the effective density corresponding to the specified deme
    size for the specified discrete model.
    """
    ret = N / 2
    if model == MODEL_MORAN:
        ret = N / 4
    return ret

def r_to_migration_rate(r=1, d=1):
    """
    Returns the stepping stone migration rate that corresponds to the
    specified event radius in the continuum model. We use
    m = sigma^2 = 2 r**2 / (d + 2)
    """
    return 2 * r**2 / (d + 2)

def migration_rate_to_r(m):
    """
    Returns the event radius in 1D that corresponds to the specified
    stepping-stone migration rate. m = sigma^2 = 2 r**2 / 3
    """
    return np.sqrt(3 * m / 2)

def subprocess_worker(args):
    """
    Function run in the subprocess to run the replicates.
    """
    sim, generations, seed = args
    sim.random_seed = seed
    sim.run_replicate(generations)
    sim.reset()

def subprocess_initialiser():
    """
    Function called when a subprocess is started.
    """
    # print("Starting process ", os.getpid())
    reset_cpu_affinity()

def run_replicates(sim, generations, num_replicates):
    """
    Runs the specified wave simulator over the specified set of times
    for the specified number of replicates.
    """
    d = sim.get_storage_directory()
    if not os.path.exists(d):
        os.makedirs(d)
    args = [(sim, generations, random.randint(1, 2**31))
            for j in range(num_replicates)]
    n = __num_worker_processes
    if n > 1:
        workers = multiprocessing.Pool(n, subprocess_initialiser)
        workers.map(subprocess_worker, args)
        workers.close()
        workers.join()
    else:
        for arg in args:
            #print("processing", arg)
            subprocess_worker(arg)

class DiscreteWaveSimulator(object):
    """
    Class representing a stepping-stone simulator for the discrete wave.
    """
    def __init__(self, num_demes, deme_size, migration_rate, num_loci,
            source_deme, max_time, output_interval, model,
            identifier="discrete"):
        self.model = model
        self.num_demes = num_demes
        self.deme_size = deme_size
        self.migration_rate = migration_rate
        self.num_loci = num_loci # 0 for a pedigree simulation
        self.max_time = max_time
        self.output_interval = output_interval
        self.source_deme = source_deme
        self.identifier = identifier
        self.generations = range(0, max_time, output_interval)
        self.executable = "./src/discrete"
        self.x = np.linspace(0, num_demes - 1, num_demes) - source_deme
        self.dimension = 1

    def set_identifier(self, identifier):
        self.identifier = identifier

    def get_storage_directory(self):
        d = "data/replicates__NOBACKUP__/{0}".format(self.identifier)
        return d

    def isdiscrete(self):
        return True

    def get_parameter_summary(self):
        """
        Returns a string describing the parameters.
        """
        return "{0}: rho_e = {1} num_loci = {2} ".format(self.model,
                self.get_effective_density(), self.num_loci)

    def write_config_file(self, f):
        """
        Writes the config file for msprime
        """
        s = ""
        s += "verbosity = 0;\n"
        s += "model = \"{0}\";\n".format(self.model)
        s += "num_demes = {0};\n".format(self.num_demes)
        s += "migration_rate = {0};\n".format(self.migration_rate)
        s += "num_loci = {0};\n".format(self.num_loci)
        s += "source_deme = {0};\n".format(self.source_deme)
        s += "deme_size = {0};\n".format(self.deme_size)
        s += "max_generations = {0};\n".format(self.max_time)
        s += "output_frequency = {0};\n".format(self.output_interval)
        s += "output_prefix = \"{0}/\";\n".format(self.get_storage_directory())
        f.write(s)
        f.flush()

    def run_replicate(self, generations):
        """
        Runs the simulator according to the parameters set at instantiation
        time. The generations parameters is ignored.
        """
        with tempfile.NamedTemporaryFile(prefix="discrete", suffix=".cnf") as f:
            self.write_config_file(f)
            seed = random.randint(1, 2**32)
            args = [self.executable, f.name, str(seed)]
            subprocess.check_call(args)


    def get_n_replicates_array(self, k):
        L = int(self.num_demes)
        return np.empty((k, L))

    def get_n_replicates(self, g):
        """
        Returns the replicates of n from the storage directory and
        and return the these as a numpy array.
        """
        d = self.get_storage_directory()
        suffix = "{0}_*.dat".format(g)
        pattern = os.path.join(d, suffix)
        files = [f for f in glob.glob(pattern)]
        N = self.get_n_replicates_array(len(files))
        for j, f in enumerate(files):
            N[j] = np.loadtxt(f)
        print("Read {0} n replicates at g={1} for {2}".format(len(files), g,
                self.get_parameter_summary()))
        return N

    def get_effective_density(self):
        """
        Returns the effective density.
        """
        return deme_size_to_rho_e(self.deme_size, self.model)

    def get_equilibrium_density(self):
        """
        Returns the equilibrium density for this simulator.
        """
        ret = self.deme_size / 2 - 1
        if self.model == MODEL_WRIGHT_FISHER:
            p = (1 + lambertw(-2 * np.exp(-2)).real / 2)
            ret = p * self.deme_size
        return ret

    def reset(self):
        pass


class WaveSimulator(discsim.Simulator):
    """
    Class representing a wave simulator.
    """
    def __init__(self, torus_diameter, simulate_pedigree, u, r=1):
        """
        Allocates a new wave simulator for the specified torus_diameter
        and neighbourhood. If simulate_pedigree is True, simulate
        pedigree ancestors.
        """
        super(WaveSimulator, self).__init__(torus_diameter, simulate_pedigree)
        self.u = u
        self.r = r
        self.event_classes = [ercs.DiscEventClass(u=u, r=r, rate=1)]
        self.sample = self.process_sample()
        self.max_occupancy = 1000 # FIXME -- needed this for r < 0.5
        self.max_population_size = self.get_max_population_size()
        self.identifier = None
        self.output_popsize = False
        # The simulator is not initialised, so we don't have access to the
        # attributes yet. We need the dimension though, so we hack to work
        # around this here.
        self.dimension = 1
        if isinstance(self.sample[1], (list, tuple)):
            self.dimension = 2
        L = torus_diameter
        self.x = np.linspace(0, L - 1, L) - L / 2

    def isdiscrete(self):
        return False

    def set_identifier(self, identifier):
        self.identifier = identifier

    def get_storage_directory(self):
        d = "data/replicates__NOBACKUP__/{0}".format(self.identifier)
        return d

    def get_parameter_summary(self):
        """
        Returns a string describing the parameters.
        """
        return "rho_e = {0} d = {1}".format(self.get_effective_density(),
                self.dimension)

    def run_replicate(self, generations):
        """
        Generate a wave for the specified list of generation numbers and
        save the state to the approriate destination.
        """
        self.run(0)
        for g in generations:
            self.run(self.generations_to_global_time(g))
            pop = self.get_population()
            self.save_replicate(g, pop)

    def save_replicate(self, g, population):
        """
        Saves the specified replicate.
        """
        if self.simulate_pedigree:
            x = population
        else:
            x = [y for y, a in population]
        n = self.calculate_n(x)
        name = "n_{0}_{1}.npy".format(g, self.random_seed)
        f = os.path.join(self.get_storage_directory(), name)
        np.save(f, n)
        if self.output_popsize:
            name = "popsize_{0}_{1}.npy".format(g, self.random_seed)
            f = os.path.join(self.get_storage_directory(), name)
            np.save(f, len(population))

    def get_n_replicates(self, g):
        """
        Returns the replicates of n from the storage directory and
        and return the these as a numpy array.
        """
        d = self.get_storage_directory()
        suffix = "n_{0}_*.npy".format(g)
        pattern = os.path.join(d, suffix)
        files = [f for f in glob.glob(pattern)]
        N = self.get_n_replicates_array(len(files))
        for j, f in enumerate(files):
            N[j] = np.load(f)
        print("Read {0} n replicates at g={1} for {2}".format(len(files), g,
                self.get_parameter_summary()))
        return N

    def get_popsize_replicates(self, g):
        """
        Returns the replicates of popsize from the storage directory and
        and return the these as a numpy array.
        """
        d = self.get_storage_directory()
        suffix = "popsize_{0}_*.npy".format(g)
        pattern = os.path.join(d, suffix)
        files = [f for f in glob.glob(pattern)]
        N = np.empty(len(files))
        for j, f in enumerate(files):
            N[j] = np.load(f)
        print("Read {0} popsize replicates at g={1} for {2}".format(len(files), g,
                self.get_parameter_summary()))
        return N

    def generations_to_global_time(self, g):
        """
        Converts the specified time in generations to global model time.
        """
        # T = g L^d / (lambda u Vd)
        # Here, lambda = 1
        L = self.torus_diameter
        d = self.dimension
        u = self.u
        r = self.r
        Vd = 2 * r if d == 1 else math.pi * r**2
        return g * L**d / (u * Vd)

    def get_effective_density(self):
        """
        Returns the effective density, 1 / (2 r u) in 1D and 1 / (pi r^2 u) in
        2D.
        """
        return u_to_rho_e(self.u, self.r, d=self.dimension)

    def get_equilibrium_density(self):
        """
        Returns the equilibrium density for the model.
        """
        p = (1 + lambertw(-2 * np.exp(-2)).real / 2)
        return 2 * self.get_effective_density() * p

class PedigreeWaveSimulator(WaveSimulator):
    """
    Class to simulate pedigree waves.
    """
    def __init__(self, torus_diameter, u, r=1):
        super(PedigreeWaveSimulator, self).__init__(torus_diameter, True, u, r)


    def get_type_string(self):
        return "pedigree"


class GeneticWaveSimulator(WaveSimulator):
    """
    Class to simulate genetic waves.
    """
    def __init__(self, torus_diameter, u, r, m, rho=0.5):
        """
        Allocates a new genetic wave simulator for the specified neighbourhood
        size, number of loci and recombination probability.
        """
        super(GeneticWaveSimulator, self).__init__(torus_diameter, False, u, r)
        self.num_loci = m
        self.recombination_probability = rho
        self.output_ancestral_material_count = False
        self.output_ancestral_block_size = False

    def get_parameter_summary(self):
        """
        Returns a string describing the parameters.
        """
        return "rho_e = {0}; m = {1}; rho = {2}".format(
                self.get_effective_density(), self.num_loci,
                self.recombination_probability)

    def get_type_string(self):
        return "genetic_m{0}_rho{1}".format(self.num_loci,
                self.recombination_probability)


    def get_blocks(self, d):
        """
        Returns the list of blocks of adjacent loci with the same ancestry
        in the specified locus-node mapping dictionary.
        """
        blocks = []
        loci = sorted(d.keys())
        m = len(loci)
        j = 0
        while j < m - 1:
            in_block = True
            b = 0
            while j < m - 1 and in_block:
                x = loci[j]
                y = loci[j + 1]
                in_block = x + 1 == y and d[x] == d[y]
                j += 1
                b += 1
            blocks.append(b)
        if m == 1:
            blocks = [0]
        blocks[-1] += 1
        assert(m == sum(blocks))
        return blocks

    def save_ancestral_material_count(self, g, population):
        """
        Saves the amount of ancestral material per ancestor within distance
        1 of each point.
        """
        a = [[] for j in range(int(self.torus_diameter))]
        for x, d in population:
            j = int(x)
            a[j].append(len(d))
            a[j + 1].append(len(d))
        name = "a_{0}_{1}.npy".format(g, self.random_seed)
        f = os.path.join(self.get_storage_directory(), name)
        np.save(f, a)

    def save_ancestral_block_size(self, g, population):
        """
        Saves the list of block sizes per ancestor within distance 1 of
        each point.
        """
        b = [[] for j in range(int(self.torus_diameter))]
        for x, d in population:
            blocks = self.get_blocks(d)
            j = int(x)
            b[j].append(blocks)
            b[j + 1].append(blocks)
        name = "b_{0}_{1}.pkl".format(g, self.random_seed)
        fname = os.path.join(self.get_storage_directory(), name)
        with open(fname, "w") as f:
            pickle.dump(b, f)


    def save_replicate(self, g, population):
        super(GeneticWaveSimulator, self).save_replicate(g, population)
        if self.output_ancestral_material_count:
            self.save_ancestral_material_count(g, population)
        if self.output_ancestral_block_size:
            self.save_ancestral_block_size(g, population)

    def get_a_replicates(self, g):
        """
        Returns the replicates of a from the storage directory and
        and return the these as a list of numpy arrays.
        """
        d = self.get_storage_directory()
        suffix = "a_{0}_*.npy".format(g)
        pattern = os.path.join(d, suffix)
        files = [f for f in glob.glob(pattern)]
        L = int(self.torus_diameter)
        A = [[] for j in range(L)]
        for f in files:
            for k, a in enumerate(np.load(f)):
                A[k].extend(a)
        print("Read {0} a replicates at g={1} for {2}".format(len(files), g,
                self.get_parameter_summary()))
        return A

    def get_block_statistics(self, g):
        """
        Processes the replicates in the storage directory and return the mean
        block length and number of blocks as a function of distance. We don't
        return the actual replicates here because the memory requirements
        are excessive.
        """
        d = self.get_storage_directory()
        suffix = "b_{0}_*.pkl".format(g)
        pattern = os.path.join(d, suffix)
        files = [f for f in glob.glob(pattern)]
        L = int(self.torus_diameter)
        total_block_length = np.zeros(L)
        total_num_blocks = np.zeros(L)
        total_individuals  = np.zeros(L)
        for fname in files:
            with open(fname, "r") as f:
                lists = pickle.load(f)
            # Each list represents an individual. Each individual is represented by a
            # list of block sizes.
            for k, individuals in enumerate(lists):
                total_individuals[k] += len(individuals)
                for individual in individuals:
                    total_num_blocks[k] += len(individual)
                    for block in individual:
                        total_block_length[k] += block
        block_length_mean = total_block_length / total_num_blocks
        num_blocks_mean = total_num_blocks / total_individuals
        print("Read {0} b replicates at g={1} for {2}".format(len(files), g,
                self.get_parameter_summary()))
        return block_length_mean, num_blocks_mean




class Simulator1D(object):
    """
    Class providing methods for processing 1D samples.
    """
    def get_max_population_size(self):
        return 6 * 10**5

    def process_sample(self):
        x = self.torus_diameter / 2
        return [None, x]

    def get_n_replicates_array(self, k):
        L = int(self.torus_diameter)
        return np.empty((k, L))

    def calculate_n(self, X):
        """
        Returns n(x) for the specified list of 1D locations.
        """
        L = int(self.torus_diameter)
        n = np.zeros(L)
        for x in X:
            j = int(x)
            n[j] += 1
            n[(j + 1) % L] += 1
        return n

class Simulator2D(object):
    """
    Class providing methods for processing 2D samples.
    """
    def get_max_population_size(self):
        return 7 * 10**5

    def process_sample(self):
        x = self.torus_diameter / 2
        return [None, (x, x)]

    def get_n_replicates_array(self, k):
        L = int(self.torus_diameter)
        return np.empty((k, L))

    def calculate_n_old(self, X):
        """
        Returns n(x) for the specified list of 2D locations.
        """
        v = self.torus_diameter / 2
        L = int(self.torus_diameter)
        n = np.zeros(L)
        for x in X:
            y = math.sqrt((x[0] - v)**2 + (x[1] - v)**2)
            j = int(y)
            n[j] += 1
            n[(j + 1) % L] += 1
        # We have counted the number of individuals in an annulus around the
        # sampling location. To normalise this back to n, we must divide by
        # area of the annulus to get the density and then multiply by pi
        # to regain a count. This works out as dividing by 4x, with a
        # special case at 0
        x = 4 * np.linspace(0, L - 1, L)
        x[0] = 1
        return n / x

    def calculate_n(self, X):
        """
        Returns n(x) for the specified list of 2D locations.
        """
        v = self.torus_diameter / 2
        L = int(self.torus_diameter)
        n = np.zeros(L)
        for x in X:
            if v - 1 <= x[1] <= v + 1:
                j = int(x[0])
                y = ((x[0] - j)**2 + (x[1] - v)**2)
                if y <= 1:
                    n[j] += 1
                j = (j + 1) % L
                y = ((x[0] - j)**2 + (x[1] - v)**2)
                if y <= 1:
                    n[j] += 1
        return n


class PedigreeWaveSimulator1D(PedigreeWaveSimulator, Simulator1D):
    """
    Class representing a simulator for the pedigree wave in 1D.
    """

class GeneticWaveSimulator1D(GeneticWaveSimulator, Simulator1D):
    """
    Class representing a simulator for the genetic wave in 1D.
    """

class PedigreeWaveSimulator2D(PedigreeWaveSimulator, Simulator2D):
    """
    Class representing a simulator for the pedigree wave in 2D.
    """

class GeneticWaveSimulator2D(GeneticWaveSimulator, Simulator2D):
    """
    Class representing a simulator for the genetic wave in 2D.
    """


def integral_equation_1d(x_array, n_array, u, ub_array=None):
    """
    Evaluates the integral equation at time t to get the estimated
    value of dn/dt.
    """
    n = interp1d(x_array, n_array)
    if ub_array is None:
        ub = lambda x: 1 - (1 - u)**n(x)
    else:
        ub = interp1d(x_array, ub_array)
    def integrand(z):
        return ub(z) * (2 - abs(x - z))
    N = len(x_array)
    dndt = np.zeros(N)
    for j in range(2, N-2):
        x = x_array[j]
        i, err = quad(integrand, x - 2, x + 2)
        dndt[j] = i - 2 * u * n(x)
    return dndt

class Figure(object):
    """
    Superclass of figures, taking care of shared details like axes
    and dimensions.
    """
    default_num_replicates = 1000

    def __init__(self, L=1000):
        self.dimension = 1
        self.L = L

    def get_K(self, u):
        """
        Returns the carrying capacity K.
        """
        const = (1 + lambertw(-2 * np.exp(-2)).real / 2)
        return (2 / u) * const

    def estimate_width_centre(self, x, n, K):
        """
        Estimate the wave parameters from the specified arrays using the
        specified value for the carrying capacity.
        """
        p = n / K
        z = np.sum(p)
        if z >=self.width_cutoff:
            p = p[int(z) - self.width_cutoff:]
        w = 4 * np.sum(p * (1 - p))
        return z, w

    def estimate_mean_wave_parameters(self, x, n_reps):
        """
        Estimates the mean parameters for the waves in the specified set
        of replicates. This estimates z and s independently for each
        replicate and returns the mean of these estimates.
        """
        num_reps = len(n_reps)
        z = np.zeros(num_reps)
        s = np.zeros(num_reps)
        x_pos = x[x >= 0]
        N = np.zeros(num_reps)
        for k, n in enumerate(n_reps):
            N[k] = n[x==0]
        K = np.mean(N)
        for k, n in enumerate(n_reps):
            n_pos = n[x >= 0]
            z[k], s[k] = self.estimate_width_centre(x_pos, n_pos, K)
        return np.mean(z), np.mean(s)


    def make_data_dir(self):
        """
        Makes a directory to hold the data files for this figure.
        """
        d = "data/{0}".format(self.identifier)
        if not os.path.exists(d):
            os.makedirs(d)

    def save_plot(self):
        """
        Saves the plot to the destination files.
        """
        files = ["figures/{0}.{1}".format(self.identifier, f) for f in ["ps", "pdf"]]
        for f in files:
            pyplot.savefig(f, dpi=72)
        pyplot.clf()

    def run_simulations(self, num_replicates):
        """
        Runs each simulator for the specified number of replicates.
        """
        for s in self.simulators:
            run_replicates(s, self.generations, num_replicates)

    def get_parameter_data_file(self, sim):
        """
        Returns the path that the data file generated by the process step
        should be stored in.
        """
        s = sim.model if sim.isdiscrete() else "continuous"
        return "data/{0}/{1}-{2}-{3}.npy".format(self.identifier,
                sim.get_effective_density(), sim.dimension, s)

    def get_plot_label(self, sim):
        """
        Returns a label to be associated with a plot line from the specified
        simulator.
        """
        rho_e = sim.get_effective_density()
        label="$\\rho_e={0}$".format(int(rho_e))
        return label

    def reset_legend_lines(self, legend):
        """
        Reset the color of the lines in the specified legend to black.
        """
        for l in legend.get_lines():
            l.set_color("black")

class PedigreeIntegralEquation1DFigure(Figure):
    """
    Class that processs the data for the integral equation figure.
    """
    identifier = "1d_pedigree_integral_equation"
    default_num_replicates = 10000

    def __init__(self):
        super(PedigreeIntegralEquation1DFigure, self).__init__()
        self.rho_e = 100
        self.u = rho_e_to_u(self.rho_e)
        self.simulator = PedigreeWaveSimulator1D(self.L, self.u)
        self.simulator.set_identifier(self.identifier)
        self.simulators = [self.simulator]
        s = self.simulator
        g = 20
        dg = 0.1
        self.generations = [g, g + dg]
        self.t = s.generations_to_global_time(g)
        self.dt = s.generations_to_global_time(dg)

    def process_data(self):
        t1 = self.t
        t2 = t1 + self.dt
        u = self.u
        s = self.simulator
        n_reps = s.get_n_replicates(self.generations[0])
        n1 = np.mean(n_reps, axis=0)
        n_reps = s.get_n_replicates(self.generations[1])
        n2 = np.mean(n_reps, axis=0)
        ub = np.mean(1 - (1 - u)**n_reps, axis=0)
        L = self.L
        dn = (n2 - n1) / self.dt
        dn1 = integral_equation_1d(self.x, n2, u) / L
        dn2 = integral_equation_1d(self.x, n2, u, ub) / L
        select = np.logical_and(self.x >= 0, n2 != 0.0)
        x = self.x[select]
        names = ["x", "sim", "ubEn", "Eubn"]
        dtype = np.dtype([(n, "float32") for n in names])
        cols = zip(self.x[select], dn[select], dn1[select], dn2[select])
        data =np.array(cols, dtype=dtype)
        f = "data/{0}.npy".format(self.identifier)
        np.save(f, data)

    def plot(self):
        datafile = "data/{0}.npy".format(self.identifier)
        d = np.load(datafile)
        pyplot.plot(d["x"], d["sim"], "-", label="Observation")
        pyplot.plot(d["x"], d["Eubn"], "--",
                label="$\\mathrm{\\mathbb{E}}[(1 - u)^N]$")
        pyplot.plot(d["x"], d["ubEn"], "-.",
            label="$(1 - u)^{\\mathrm{\\mathbb{E}[N]}}$")
        pyplot.legend()
        pyplot.ylabel("$\\partial n/ \\partial t$")
        pyplot.xlabel("$x$")
        pyplot.ylim(0)
        self.save_plot()


class PedigreeNumericsComparison1DFigure(Figure):
    """
    Class that processs the data for the figure comparing numerical
    results for the pedigree wave with simulations and the logistic
    approximation.
    """
    identifier = "1d_pedigree_numerics_comparison"
    default_num_replicates = 10000

    def __init__(self):
        super(PedigreeNumericsComparison1DFigure, self).__init__()
        self.rho_e = 100
        self.u = rho_e_to_u(self.rho_e)
        self.simulator = PedigreeWaveSimulator1D(self.L, self.u)
        self.simulator.set_identifier(self.identifier)
        self.simulators = [self.simulator]
        s = self.simulator
        self.generations = [4, 32, 64]
        self.times = [s.generations_to_global_time(g) for g in self.generations]
        self.numerics_cache = "data/{0}.numerical_cache.npy".format(self.identifier)

    def solve_numerics(self):
        """
        Returns the predicted numerical values for the specified
        list of times.
        """
        Lx = 300.0
        u = self.u
        dt = 0.5
        mesh = fipy.Grid1D(nx=1200, Lx=Lx)
        n = fipy.CellVariable(name="n",  mesh=mesh, value=0., hasOld=True)
        n.constrain(0, mesh.facesRight)
        n.constrain(0, mesh.facesLeft)
        X = mesh.cellCenters
        n.setValue(1.0, where=(X>=Lx/2 - 1) & (X<=Lx/2 + 1))
        D = 4 * u * fipy.numerix.exp(-n * u) / 3.0
        st = 4 - 4 * fipy.numerix.exp(-n * u) - 2 * u * n
        eq = fipy.TransientTerm() == st + fipy.DiffusionTerm(coeff=D)
        viewer = fipy.Viewer(vars=n, datamin=0., datamax=(2 / u) * 0.8,
                xmin=Lx / 2, xmax=Lx)
        viewer.plot()
        desiredResidual = 1e-4
        t = 0.0
        output = []
        j = 0
        # Divide the times by L to rescale.
        times = np.array(self.times) / self.L
        print("Solving PDE at", times)
        while j < len(times):
            n.updateOld()
            residual = 1000
            while residual > desiredResidual:
                residual = eq.sweep(var=n, dt=dt)
            if times[j] == t:
                output.append(np.array(n))
                print(t)
                j += 1
            t += dt
            if int(t) % 10 == 0:
                viewer.plot()
        x = np.array(X[0]) - Lx / 2
        return x, output

    def get_numerics(self):
        """
        Returns the numerical solutions. If the values are cached,
        return them. Otherwise, run the FiPy solver/
        """
        if os.path.exists(self.numerics_cache):
            x, n = np.load(self.numerics_cache)
        else:
            x, n = self.solve_numerics()
            np.save(self.numerics_cache, (x, n))
        return x, n


    def process_data(self):
        self.make_data_dir()
        x_numeric, n_numeric = self.get_numerics()
        max_x = 200
        names = ["x", "n_expected", "n_numerical"]
        dtype = np.dtype([(n, "float32") for n in names])
        for j, (g, t) in enumerate(zip(self.generations, self.times)):
            n_reps = self.simulator.get_n_replicates(g)
            select = np.logical_and(self.x >= 0, self.x <= max_x)
            x = self.x[select]
            mean_n = np.mean(n_reps, axis=0)
            f = interp1d(self.x, mean_n, bounds_error=False, fill_value=0)
            n_expected = f(x)
            f = interp1d(x_numeric, n_numeric[j], bounds_error=False,
                    fill_value=0)
            cols = zip(x, n_expected, f(x))
            data = np.array(cols, dtype=dtype)
            f = "data/{0}/{1}.npy".format(self.identifier, g)
            np.save(f, data)

    def plot(self):
        fig = pyplot.figure()
        ax = fig.add_subplot(111, xlim=(0,140), ylim=(1e-1,1e3))
        for g, t in zip(self.generations, self.times):
            f = "data/{0}/{1}.npy".format(self.identifier, g)
            d = np.load(f)
            x = d["x"]
            n_expected = d["n_expected"]
            n_numerical = d["n_numerical"]
            l_expected , = ax.plot(x, n_expected)
            l_numerical, = ax.plot(x, n_numerical, "--",
                    color=l_expected.get_color())
            v = np.logspace(math.log10(1e-1), math.log10(max(n_numerical)),
                    num=3)[1]
            j = (np.abs(n_numerical - v)).argmin()
            xy = x[j], n_numerical[j]
            ax.annotate("$g={0}$".format(g), xy=xy,  xycoords='data',
                    xytext=(5, 0), textcoords='offset points')
        pyplot.yscale("log")
        pyplot.ylabel("$n(x)$")
        pyplot.xlabel("$x$")
        leg = ax.legend([l_expected, l_numerical],
                ["Continuous",  "PDE"])
        self.reset_legend_lines(leg)
        self.save_plot()


class WaveParametersFigure(Figure):
    """
    Class containing the common code required to process files and generate
    estimates for the wave parameters.
    """
    def get_colour_key(self, sim):
        return 0

    def plot_model_legend(self):
        return True

    def process_data(self):
        self.make_data_dir()
        names = ["g", "z", "w"]
        dtype = np.dtype([(n, "float32") for n in names])
        L = self.L
        for sim in self.simulators:
            z = np.zeros(len(self.generations))
            w = np.zeros(len(self.generations))
            for j, g in enumerate(self.generations):
                reps = sim.get_n_replicates(g)
                z[j], w[j] = self.estimate_mean_wave_parameters(sim.x, reps)
            g = np.array(self.generations)
            cols = zip(g, z, w)
            data = np.array(cols, dtype=dtype)
            f = self.get_parameter_data_file(sim)
            np.save(f, data)

    def plot(self):
        fig, (ax1, ax2) = pyplot.subplots(2, sharex=True)
        fig.set_size_inches(8, 8)
        self.colours = {}
        for sim in self.simulators:
            f = self.get_parameter_data_file(sim)
            d = np.load(f)
            lbl = self.get_plot_label(sim)
            if sim.isdiscrete():
                m = "--" if sim.model == MODEL_WRIGHT_FISHER else ":"
                l, = ax1.plot(d["g"], d["z"], m,
                        color=self.colours[self.get_colour_key(sim)])
            else:
                m = "-"
                l, = ax1.plot(d["g"], d["z"], m, label=lbl)
                self.colours[self.get_colour_key(sim)] = l.get_color()
            ax2.plot(d["g"], d["w"], m, color=l.get_color(), label=lbl)
        if self.plot_model_legend():
            # add in proxy artists for the marker legends
            lines = []
            markers = [("Moran", ":"), ("Continuum", "-"), ("Wright-Fisher", "--")]
            for model, marker in markers:
                l, = ax2.plot(d["g"], d["w"], marker, color="black")
                lines.append(l)
                l.remove()
            ax2.legend(lines, [s for (s,m) in markers], loc="lower right")
        ax1.set_ylabel("Wave centre")
        ax2.set_ylabel("Wave width")
        ax2.set_xlabel("Generations")
        ax1.legend(loc="upper left")
        self.tweak_plot(ax1, ax2)
        self.save_plot()


class PedigreeWaveParameters1DFigure(WaveParametersFigure):
    """
    Class that processs the data for the figure plotting the wave parameters
    over time.
    """
    identifier = "1d_pedigree_wave_parameters"

    def __init__(self):
        super(PedigreeWaveParameters1DFigure, self).__init__(1000)
        self.width_cutoff = 15
        self.rho_e = [75, 100, 125]
        max_generation = 100
        gap = 5
        self.generations = np.arange(0, max_generation + 1, gap)
        self.simulators = []
        m = 0.5
        r = migration_rate_to_r(m)
        for rho_e in self.rho_e:
            s = PedigreeWaveSimulator1D(self.L, rho_e_to_u(rho_e, r), r)
            s.set_identifier(self.identifier + "/continuous_{0}".format(rho_e))
            s.discrete_migration_rate = m
            self.simulators.append(s)
            for model in [MODEL_WRIGHT_FISHER, MODEL_MORAN]:
                idf = self.identifier + "/{0}_{1}".format(model, rho_e)
                N = rho_e_to_deme_size(rho_e, model)
                s = DiscreteWaveSimulator(self.L, N, m, 0, 100, max_generation,
                        gap, model, idf)
                self.simulators.append(s)

    def tweak_plot(self, ax_z, ax_w):
        ax_w.set_ylim(0)
        #ax_z.set_ylim(-5, 130)
        #ax_z.set_xlim(0, 101)

class GeneticWaveParametersFigure(WaveParametersFigure):
    """
    Superclass of Genetic wave parameters figure. Takes care of plotting the
    overlaid fit to sqrt(t).
    """
    def get_colour_key(self, sim):
        return sim.num_loci

    def get_parameter_data_file(self, sim):
        s = sim.model if sim.isdiscrete() else "continuous"
        return "data/{0}/{1}.{2}.npy".format(self.identifier, s, sim.num_loci)

    def get_plot_label(self, sim):
        lm = int(round(math.log(sim.num_loci, 10)))
        return "$10^{0}$ loci".format(lm)

    def tweak_plot(self, ax_z, ax_w):

        def func(t, a):
            return a * np.sqrt(t)

        for sim in self.simulators:
            if not sim.isdiscrete():
                f = self.get_parameter_data_file(sim)
                d = np.load(f)
                z = np.array(d["z"], dtype=np.float64)
                g = np.array(d["g"], dtype=np.float64)
                popt, pcov = curve_fit(func, g, z)
                col = self.colours[self.get_colour_key(sim)]
                ax_z.plot(g, func(g, popt[0]), "o", color=col)




class GeneticWaveParameters1DFigure(GeneticWaveParametersFigure):
    """
    Class that processs the data for the figure comparing the wave
    parameters for the genetic wave in 1D.
    """
    identifier = "1d_genetic_wave_parameters"

    def __init__(self):
        super(GeneticWaveParameters1DFigure, self).__init__()
        self.rho_e = 100
        self.width_cutoff = 1000
        self.migration_rate = 0.5
        self.r = migration_rate_to_r(self.migration_rate)
        self.u = rho_e_to_u(self.rho_e, r=self.r)
        self.m = [10**3, 10**4, 10**5]
        self.simulators = []
        gap = 25
        max_generation = 500
        self.generations = np.arange(0, max_generation + 1, gap)
        for m in self.m:
            s = GeneticWaveSimulator1D(self.L, self.u, self.r, m)
            idf = self.identifier + "/continuous.{0}".format(s.num_loci)
            s.set_identifier(idf)
            self.simulators.append(s)
            for model in [MODEL_WRIGHT_FISHER, MODEL_MORAN]:
                idf = self.identifier + "/{0}.{1}".format(model, s.num_loci)
                N = rho_e_to_deme_size(self.rho_e, model)
                s = DiscreteWaveSimulator(self.L, N, self.migration_rate, m, 100,
                        max_generation, gap, model, idf)
                self.simulators.append(s)

    def tweak_plot(self, ax_z, ax_w):
        super(GeneticWaveParameters1DFigure, self).tweak_plot(ax_z, ax_w)
        ax_z.set_ylim(1, 50)


class GeneticWaveParameters2DFigure(GeneticWaveParametersFigure):
    """
    2D genetic wave paramters.
    """
    identifier = "2d_genetic_wave_parameters"

    def __init__(self):
        super(GeneticWaveParameters2DFigure, self).__init__()
        self.rho_e = 100
        self.width_cutoff = 1000
        self.u = rho_e_to_u(self.rho_e, d=2)
        self.m = [10**3, 10**4, 10**5]
        self.simulators = []
        gap = 10
        max_generation = 200
        self.generations = np.arange(0, max_generation + 1, gap)
        for m in self.m:
            s = GeneticWaveSimulator2D(self.L, self.u, 1, m)
            s.max_occupancy = 5000
            idf = self.identifier + "/{0}".format(s.num_loci)
            s.set_identifier(idf)
            self.simulators.append(s)

    def plot_model_legend(self):
        return False

    def tweak_plot(self, ax_z, ax_w):
        super(GeneticWaveParameters2DFigure, self).tweak_plot(ax_z, ax_w)
        ax_z.set_ylim(2)

class PedigreeGeneticComparison1DFigure(Figure):
    """
    Class that processs the data for the figure comparing the wave
    of pedigree and genetic ancestors in 1D.
    """
    identifier = "1d_pedigree_genetic_comparison"

    def __init__(self):
        super(PedigreeGeneticComparison1DFigure, self).__init__()
        self.rho_e = 100
        self.u = rho_e_to_u(self.rho_e)
        self.pedigree_simulator = PedigreeWaveSimulator1D(self.L, self.u)
        self.pedigree_simulator.set_identifier(self.identifier + "/pedigree")
        self.genetic_simulator = GeneticWaveSimulator1D(self.L, self.u, 1, 10**5)
        self.genetic_simulator.set_identifier(self.identifier + "/genetic")
        self.simulators = [self.pedigree_simulator, self.genetic_simulator]
        self.generations = [5, 10, 20]

    def process_data(self):
        self.make_data_dir()
        names = ["x", "n"]
        dtype = np.dtype([(n, "float32") for n in names])
        for j, g in enumerate(self.generations):
            for sim in [self.pedigree_simulator, self.genetic_simulator]:
                n_reps = sim.get_n_replicates(g)
                mean_n = np.mean(n_reps, axis=0)
                select = np.logical_and(self.x >= 0, mean_n > 0.0)
                x = self.x[select]
                n = mean_n[select]
                """
                z, s = self.estimate_mean_wave_parameters(self.x, n_reps,
                        sim.u)
                y, n = self.get_centred_mean_wave(n_reps)
                x = z + y
                select = np.logical_and(x >= 0, x < 50)
                cols = zip(x[select], n[select])
                """
                cols = zip(x, n)
                data = np.array(cols, dtype=dtype)
                f = "data/{0}/{1}_{2}.npy".format(self.identifier,
                        sim.get_type_string(), g)
                np.save(f, data)

    def plot(self):
        fig = pyplot.figure()
        ax = fig.add_subplot(111, xlim=(0,30))
        for g in self.generations:
            sim = self.pedigree_simulator
            f = "data/{0}/{1}_{2}.npy".format(self.identifier,
                    sim.get_type_string(), g)
            d = np.load(f)
            x = d["x"]
            n = d["n"]
            lp, = ax.plot(x, n)
            j = (np.abs(n- max(n) / 2)).argmin()
            xy = x[j], n[j]
            ax.annotate("$g={0}$".format(g), xy=xy,  xycoords='data',
                    xytext=(5, 0), textcoords='offset points')
            sim = self.genetic_simulator
            f = "data/{0}/{1}_{2}.npy".format(self.identifier,
                    sim.get_type_string(), g)
            d = np.load(f)
            lg, = ax.plot(d["x"], d["n"], "--", color=lp.get_color())
        pyplot.ylabel("$n(x)$")
        pyplot.xlabel("$x$")
        leg = ax.legend([lp, lg], ["Pedigree", "Genetic"])
        self.reset_legend_lines(leg)
        self.save_plot()

class Pedigree1D2DParametersComparisonFigure(WaveParametersFigure):
    """
    Class that processs the data for the figure comparing the wave
    parameters for the  pedigree ancestors in 1 and 2D.
    """
    identifier = "1d_2d_pedigree_parameters_comparison"

    def __init__(self):
        super(Pedigree1D2DParametersComparisonFigure, self).__init__()
        self.width_cutoff = 15
        self.rho_e = 100
        self.u_1d = rho_e_to_u(self.rho_e, d=1)
        self.u_2d = rho_e_to_u(self.rho_e, d=2)
        self.simulator_1d = PedigreeWaveSimulator1D(self.L, self.u_1d)
        self.simulator_1d.set_identifier(self.identifier + "/1d")
        self.simulator_2d_rho = PedigreeWaveSimulator2D(self.L, self.u_2d)
        self.simulator_2d_rho.set_identifier(self.identifier + "/2d_rho")
        self.simulator_2d_u = PedigreeWaveSimulator2D(self.L, self.u_1d)
        self.simulator_2d_u.set_identifier(self.identifier + "/2d_u")
        self.generations = np.linspace(1, 30, num=20)
        self.simulators = [self.simulator_1d, self.simulator_2d_rho,
                self.simulator_2d_u]


    def get_plot_label(self, sim):
        rho_e = u_to_rho_e(sim.u, d=sim.dimension)
        if int(rho_e) == rho_e:
            label="${0}\\mathrm{{D}}:\\rho_e={1}$".format(sim.dimension, int(rho_e))
        else:
            label="${0}\\mathrm{{D}}:\\rho_e={1:.2f}$".format(sim.dimension, rho_e)
        return label

    def plot_model_legend(self):
        return False

    def tweak_plot(self, ax_z, ax_w):
        ax_z.set_ylim(1, 30)

class AncestralMaterialLinearGenome1DFigure(Figure):
    """
    Class that processs the data for the figure showing the
    the block size and number of blocks as a function of distance.
    """
    identifier = "1d_ancestral_material_linear_genome"
    default_num_replicates = 100000

    def __init__(self):
        super(AncestralMaterialLinearGenome1DFigure, self).__init__()
        self.rho_e = 100
        self.u = rho_e_to_u(self.rho_e)
        self.m = 10**5
        self.rho = 1e-3
        self.simulator = GeneticWaveSimulator1D(self.L, self.u, self.m,
                self.rho)
        self.simulator.set_identifier(self.identifier)
        self.simulator.output_ancestral_block_size = True
        self.simulators = [self.simulator]
        self.generations = [100]

    def process_data(self):
        self.make_data_dir()
        L = int(self.simulator.torus_diameter)
        names = ["x", "n", "bl_mean", "nb_mean"]
        dtype = np.dtype([(n, "float32") for n in names])
        g = self.generations[0]
        n_reps = self.simulator.get_n_replicates(g)
        mean_n = np.mean(n_reps, axis=0)
        bl_mean, nb_mean = self.simulator.get_block_statistics(g)
        select = np.logical_and(self.x >= 0, mean_n > 0.0)
        x = self.x[select]
        n = mean_n[select]
        bl_mean = bl_mean[select]
        nb_mean = nb_mean[select]
        cols = zip(x, n, bl_mean, nb_mean)
        data = np.array(cols, dtype=dtype)
        f = "data/{0}/{1}.npy".format(self.identifier, g)
        np.save(f, data)

    def plot(self):
        g = self.generations[0]
        f = "data/{0}/{1}.npy".format(self.identifier, g)
        d = np.load(f)
        x = d["x"]
        n = d["n"]
        pyplot.plot(x, n, label="n(x)", color="g", linestyle="--")
        pyplot.plot(x, d["bl_mean"], label="block length", color="r",
                linestyle="-.")
        pyplot.plot(x, d["nb_mean"], label="num blocks", color="b")
        pyplot.legend()
        pyplot.axhline(1, color="black", linestyle="--")
        #pyplot.ylim(1e-1, 1e3)
        pyplot.ylim(0, 15)
        pyplot.xlim(0, 40)
        #pyplot.yscale("log")
        pyplot.xlabel("$x$")
        self.save_plot()


class PedigreeClineShape1DFigure(Figure):
    """
    Figure showing the cline shape for the different model along
    with the theoretical predictions for this.
    """
    default_num_replicates = 1000
    data_identifier = "1d_pedigree_cline_shape"

    def __init__(self):
        super(PedigreeClineShape1DFigure, self).__init__(1000)
        g = 20
        self.generations = [g]
        m = 0.5
        self.sigma = np.sqrt(m)
        gap = 5
        self.width_cutoff = 10
        rho_e = 100
        self.simulators = []
        r = migration_rate_to_r(m)
        s = PedigreeWaveSimulator1D(self.L, rho_e_to_u(rho_e, r), r)
        s.set_identifier(self.data_identifier + "/Continuum")
        s.discrete_migration_rate = m
        self.simulators.append(s)
        for model in [MODEL_MORAN, MODEL_WRIGHT_FISHER]:
            idf = self.data_identifier + "/{0}".format(model)
            N = rho_e_to_deme_size(rho_e, model)
            s = DiscreteWaveSimulator(self.L, N, m, 0, 100, g,
                    g, model, idf)
            self.simulators.append(s)


    def process_data(self):
        # Crude hack to get the data directory in the right place
        tmp = self.identifier
        self.identifier = self.data_identifier
        self.make_data_dir()
        self.identifier = tmp
        g = self.generations[0]
        window = 20 # The width of the aggregated wave
        cols = {}
        for s in self.simulators:
            x = s.x
            x_pos = x[x >= 0]
            K = s.get_equilibrium_density()
            n_reps = s.get_n_replicates(g)
            C = np.zeros((len(n_reps), window))
            if not s.isdiscrete():
                n_reps = n_reps / 2
            for j, n in enumerate(n_reps):
                n_pos = n[x >= 0]
                z, w = self.estimate_width_centre(x_pos, n_pos, K)
                y = s.x - z
                n = n / K
                mask = (-window / 2 < y) & (y <= window / 2)
                C[j] = n[mask]
            f = "data/{0}.npy".format(s.identifier)
            np.save(f, C)
            cols[s.identifier.split("/")[1]] = np.mean(C, axis=0)
        y = np.arange(-window / 2 + 0.5, window / 2 + 0.5)
        cols["x"] = y
        # Add in the Gaussian approx
        rho_sigma = s.get_effective_density() * self.sigma
        z = position_gauss(g, rho_sigma)
        cols["Gaussian"] = gaussian_approximation(y + z, g, rho_sigma)

        dtype = np.dtype([(n, "float32") for n in cols.keys()])
        data = np.array(zip(*cols.values()), dtype=dtype)
        f = "data/{0}/summary.npy".format(self.data_identifier)
        np.save(f, data)


class PedigreeClineShape1DReplicatesFigure(PedigreeClineShape1DFigure):
    """
    Figure showing the cline shape for the different model in raw replicate
    form.
    """
    identifier = "1d_pedigree_cline_shape_replicates"

    def plot(self):

        f = "data/{0}/x.npy".format(self.data_identifier)
        x = np.load(f)
        fig, ax = pyplot.subplots(1, 3, sharey=True)
        fig.set_size_inches(16, 4)

        for j, s in enumerate(self.simulators):
            f = "data/{0}.npy".format(s.identifier)
            P = np.load(f)
            for p in P:
                ax[j].plot(x, p)
            p = np.mean(P, axis=0)
            ax[j].plot(x, p, "--", linewidth=5, color="black")
            title = s.identifier.split("/")[1].title()
            ax[j].set_title(title)
        ax[0].set_ylim(-0.05, 1.2)
        ax[0].set_ylabel("$p$")
        self.save_plot()

class PedigreeClineShape1DMeanFigure(PedigreeClineShape1DFigure):
    """
    Figure showing the mean cline shape for the different models
    along with the various theoretical predictions
    """
    identifier = "1d_pedigree_cline_shape_mean"

    def plot(self):

        f = "data/{0}/summary.npy".format(self.data_identifier)
        d = np.load(f)
        x = d["x"]
        models = ["Continuum", "Wright-Fisher", "Moran", "Gaussian"]
        lines = ["-", "--", ":", "-."]
        for m, l in zip(models, lines):
            p = d[m]
            pyplot.plot(x, 1 / (1 / p - 1), l, label=m)

        # Get the Fisher-Wave solution
        def f(p, y):
            return [-2 - p * (1 - p) / y[0], 1 / y[0]]
        r = ode(f)
        y0 = [-0.0009, 0]
        r.set_initial_value(y0, 0.999)
        dp = -0.001
        P = [0.999]
        X = [y0[1]]
        while r.successful() and r.t > 0.001:
            r.integrate(r.t + dp)
            P.append(r.t)
            X.append(r.y[1])
        p = np.array(P)
        x = np.array(X)
        # Find the centre of this curve z
        f = interp1d(x, p, bounds_error=False)
        z, err = quad(f, min(x), max(x), limit=100)
        pyplot.plot((x - z) / np.sqrt(2 * np.log(2)), 1 / (1 / f(x) - 1), "-k",
                label="Fisher-KPP")
        pyplot.yscale("log")
        pyplot.xlim(-10, 10)
        pyplot.ylim(1e-3, 1e3)
        pyplot.ylabel("$\\frac{1}{\\frac{1}{p} -1}$")
        pyplot.legend()
        self.save_plot()


def run_simulate(cls, args):
    global __num_worker_processes
    __num_worker_processes = args.p
    f = cls()
    n = cls.default_num_replicates if args.n == -1 else args.n
    f.run_simulations(n)

def run_process(cls, args):
    f = cls()
    f.process_data()

def run_plot(cls, args):
    f = cls()
    f.plot()
def run_list(keys):
    print("Available plots:")
    for k in keys:
        print("\t", k)

def main():
    reset_cpu_affinity()

    plots = [
        PedigreeIntegralEquation1DFigure,
        PedigreeNumericsComparison1DFigure,
        PedigreeWaveParameters1DFigure,
        GeneticWaveParameters1DFigure,
        GeneticWaveParameters2DFigure,
        PedigreeGeneticComparison1DFigure,
        Pedigree1D2DParametersComparisonFigure,
        # TODO fix this bug and reenable.
        # AncestralMaterialLinearGenome1DFigure,
        PedigreeClineShape1DReplicatesFigure,
        PedigreeClineShape1DMeanFigure,
    ]
    key_map = dict([(c.identifier, c) for c in plots])
    num_cpus = multiprocessing.cpu_count()
    parser = argparse.ArgumentParser(description=
            "Run simulations and process data files.")
    subparsers = parser.add_subparsers()
    simulate_parser = subparsers.add_parser('simulate')
    simulate_parser.add_argument('-p', help="number of processors",
            type=int, default=num_cpus)
    simulate_parser.add_argument('-n', help="number of replicates",
            type=int, default=-1)
    simulate_parser.add_argument('key', metavar='KEY', type=str, nargs=1,
            help='the simulation identifier')
    simulate_parser.set_defaults(func=run_simulate)

    process_parser = subparsers.add_parser('process')
    process_parser.add_argument('key', metavar='KEY', type=str, nargs=1,
            help='the simulation identifier')
    process_parser.set_defaults(func=run_process)

    plot_parser = subparsers.add_parser('plot')
    plot_parser.add_argument('key', metavar='KEY', type=str, nargs=1,
            help='the simulation identifier')
    plot_parser.set_defaults(func=run_plot)

    list_parser = subparsers.add_parser("list")
    list_parser.set_defaults(func=run_list)

    args = parser.parse_args()
    if args.func == run_list:
        run_list(key_map.keys())
    else:
        k = args.key[0]
        if k == "all":
            for key, cls in key_map.items():
                print("Running:", key)
                args.func(cls, args)
        else:
            cls = key_map[k]
            args.func(cls, args)


if __name__ == "__main__":
    main()
