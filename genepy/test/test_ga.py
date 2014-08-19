import math
import numpy
import random
import logging
import pprint

from genepy import ga
from genepy import generate
from genepy import mutate
from genepy import crossover

_logger = logging.getLogger(__name__)

def test_search():


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

    def _fitness(genes):
        a_error = math.fabs(genes['a'] - target['a']) / target['a_range']
        b_error = math.fabs(genes['b'] - target['b']) / target['b_range']
        return 1.0 - (a_error + b_error)

    kwargs = {
        'max_iterations': 50,
        'best_ratio_thresh': 0.01,
        'active_genes': ['a', 'b'],
        'mixing_ratio': 0.5,
        'num_replaces': 1
    }
    def _update_mutation(
        iterations, 
        individuals, 
        genepool, 
        gene_properties, 
        fitness, 
        **kwargs):

        for gene in kwargs['active_genes']:
            gene_parameters[gene]['mutate'] = mutate.population_gaussian(
                individuals, 
                genepool, 
                gene)

    def _print_population(
        iterations, 
        individuals, 
        genepool, 
        gene_properties, 
        fitness, 
        **kwargs):

        pprint.pprint(individuals)


    result = ga.search(
        population_size,
        gene_parameters,
        _fitness,
        ga.create_generation_callstack([
            ga.max_iteration_convergence,
            ga.best_ratio_convergence,
            _update_mutation,
            _print_population
        ]),
        **kwargs)

    assert(not result is None)
    
    _logger.info('result {0}'.format(result))
    _logger.info('target {0}'.format(target))
    _logger.info('fitness of result {0}'.format(_fitness(result)))
    assert(_fitness(result) >= 0.99)



    


