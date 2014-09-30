import math
import numpy
import random
import logging
import pprint
import time
import multiprocessing

from genepy import ga
from genepy import generate
from genepy import mutate
from genepy import crossover

_logger = logging.getLogger(__name__)

def _create_fitness(target):
    def _fitness(genes):
        a_error = math.fabs(genes['a'] - target['a']) / target['a_range']
        b_error = math.fabs(genes['b'] - target['b']) / target['b_range']
        return 1.0 - (a_error + b_error)
    return _fitness

class SleepyFitness(object):
    def __init__(self, target, sleep_time):
        self.target = target
        self.sleep_time = sleep_time

    def __call__(self, genes):
        time.sleep(self.sleep_time)
        a_error = math.fabs(genes['a'] - self.target['a']) / self.target['a_range']
        b_error = math.fabs(genes['b'] - self.target['b']) / self.target['b_range']
        return 1.0 - (a_error + b_error)

def _fitness(target, genes):
    f = _create_fitness(target)
    return f(genes)

def _run_search_with_kwargs(kwargs, fitness):
    # just try to guess two numbers within some distance
    population_size = 10
    gene_parameters = {
        'a': {
            'generate': generate.sample(random.uniform, -10.0, 10.0),
            'mutate': mutate.gaussian(0.5),
            'crossover': crossover.sample_gaussian,
        },
        'b': {
            'generate': generate.sample(random.uniform, -100.0, 100.0),
            'mutate': mutate.gaussian(1.0),
            'crossover': crossover.sample_gaussian,
        }
    }
    def _update_mutation(
        iterations, 
        individuals, 
        genepool, 
        gene_properties, 
        fitness, 
        **kwargs):

        for gene in gene_parameters.iterkeys():
            gene_parameters[gene]['mutate'] = mutate.population_gaussian(
                individuals, 
                genepool, 
                gene)

    result = ga.search(
        population_size,
        gene_parameters,
        fitness,
        ga.create_generation_callstack([
            ga.max_iteration_convergence,
            ga.best_ratio_convergence,
            _update_mutation,
        ]),
        **kwargs)
    return result




def test_ga_timeout():
    kwargs = {
        'max_iterations': 50,
        'best_ratio_thresh': 0.01,
        'mixing_ratio': 0.1,
        'num_replaces': 10,
        'processor_count': 4,
        #'timeout': 0.1,
        #'timeout_fitness': -1
    }
    target = {
        'a': numpy.random.uniform(-10.0, 10.0),
        'b': numpy.random.uniform(-100.0, 100.0),
        'a_range': 20.0,
        'b_range': 200.0
    }
    fit = SleepyFitness(target, 0.01)
    try:
        result = _run_search_with_kwargs(kwargs, fit)
    except:
        assert False
    kwargs = {
        'max_iterations': 50,
        'best_ratio_thresh': 0.01,
        'mixing_ratio': 0.1,
        'num_replaces': 10,
        'processor_count': 4,
        'timeout': 0.001,
        #'timeout_fitness': -1
    }
    try:
        result = _run_search_with_kwargs(kwargs, fit)
    except multiprocessing.TimeoutError, e:
        pass
    else:
        assert False
    kwargs = {
        'max_iterations': 50,
        'best_ratio_thresh': 0.01,
        'mixing_ratio': 0.1,
        'num_replaces': 10,
        'processor_count': 4,
        'timeout': 0.001,
        'timeout_fitness': -1
    }
    try:
        result = _run_search_with_kwargs(kwargs, fit)
    except:
        assert False
    
if __name__ == '__main__':
    test_ga_timeout()

