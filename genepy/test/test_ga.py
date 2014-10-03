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
from genepy import population

_logger = logging.getLogger(__name__)



def _create_fitness(target):
    def _fitness(genes):
        a_error = math.fabs(genes['a'] - target['a']) / target['a_range']
        b_error = math.fabs(genes['b'] - target['b']) / target['b_range']
        return 1.0 - (a_error + b_error)
    return _fitness

def _fitness(target, genes):
    f = _create_fitness(target)
    return f(genes)

def _run_search_with_kwargs(kwargs):
    # just try to guess two numbers within some distance
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
    assert(_fitness(target, result) >= 0.99)

    kwargs = {
        'max_iterations': 100,
        'best_ratio_thresh': 0.001,
        'mixing_ratio': 0.4,
        'num_replaces': 3
    }
    target, result = _run_search_with_kwargs(kwargs)
    assert(_fitness(target, result) >= 0.99)

prexist_call_count = 0
def test_prexisting_genepool():
    global prexist_call_count
    prexist_call_count = 0
    genes = {
        'a': 5,
        'b': 10
    }

    individuals = [population.gene_hash_function(genes)]
    genepool = {individuals[0] : genes}
    fitness = {individuals[0] : 10}

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

    def on_generation(
        iterations, 
        individuals, 
        genepool, 
        gene_properties, 
        fitness, 
        **kwargs):
        global prexist_call_count
        prexist_call_count += 1
        
        assert(len(individuals) == 1)
        assert(len(genepool) == 1)
        assert(genepool[individuals[0]]['a'] == 5)
        assert(genepool[individuals[0]]['b'] == 10)
        assert(fitness[individuals[0]] == 10)

        return individuals[0]

    def _fitness(x):
        return -1

    result = ga.search(
        100,
        gene_parameters,
        _fitness,
        on_generation,
        individuals=individuals,
        genepool=genepool,
        fitness=fitness)

    assert(result == individuals[0])
    assert(prexist_call_count == 1)




call_count = 0
def test_fitness_cache():
    global call_count

    def _gen():
        x = 1
        while True:
            x = x + 1
            if x > 5:
                x = 1
            yield x

    call_count = 0
    def _call_count_fitness(genes):
        global call_count
        call_count += 1
        return genes['a']

    gene_props = {
        'a': {
            'generate': _gen(),
            'mutate': lambda x: x,
            'crossover': crossover.swap
        }
    }

    kwargs = {
        'max_iterations': 10,
    }


    result = ga.search(
        500,
        gene_props,
        _call_count_fitness,
        ga.create_generation_callstack([
            ga.max_iteration_convergence,
        ]),
        **kwargs)

    assert(call_count == 5)
    assert(not (result is None))
    assert(result['a'] == 5)





