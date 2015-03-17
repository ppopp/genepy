import math
import numpy
import random
import logging
import pprint
import time

from genepy import ga
from genepy import generate
from genepy import mutate
from genepy import crossover

_logger = logging.getLogger(__name__)



def _create_fitness(target):
    def _fitness(genepool):
        fitness = {}
        for individual, genes in genepool.iteritems():
            a_error = math.fabs(genes['a'] - target['a']) / target['a_range']
            b_error = math.fabs(genes['b'] - target['b']) / target['b_range']
            fitness[individual] = 1.0 - (a_error + b_error)
        return fitness
    return _fitness

def _fitness(target, genes):
    f = _create_fitness(target)
    return f({'best':genes})['best']

def _run_search_with_kwargs(kwargs):
    # just try to guess two numbers within some distance
    kwargs['batch'] = True
    population_size = 51
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
    target = {
        'a': numpy.random.uniform(-10.0, 10.0),
        'b': numpy.random.uniform(-100.0, 100.0),
        'a_range': 20.0,
        'b_range': 200.0
    }
    def _update_mutation(
        iterations, 
        individuals, 
        genepool, 
        gene_properties, 
        fitness, 
        **kwargs):

        for gene in gene_properties.keys():
            gene_parameters[gene]['mutate'] = mutate.population_gaussian(
                individuals, 
                genepool, 
                gene)

    result = ga.search(
        population_size,
        gene_parameters,
        _create_fitness(target),
        ga.create_generation_callstack([
            ga.max_iteration_convergence,
            ga.best_ratio_convergence,
            _update_mutation,
        ]),
        **kwargs)
    return (target, result)


def test_search():
    kwargs = {
        'max_iterations': 50,
        'best_ratio_thresh': 0.001,
        'active_genes': ['a', 'b'],
        'mixing_ratio': 0.5,
        'num_replaces': 1
    }
    target, result = _run_search_with_kwargs(kwargs)
    assert(_fitness(target, result) >= 0.9)

    kwargs = {
        'max_iterations': 100,
        'best_ratio_thresh': 0.001,
        'mixing_ratio': 0.4,
        'num_replaces': 3
    }
    target, result = _run_search_with_kwargs(kwargs)
    assert(_fitness(target, result) >= 0.9)






