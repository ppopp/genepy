import operator
import logging

import numpy

import population

_logger = logging.getLogger(__name__)

def _fitness_of_population(fitness, population=None):
    results_in_population = None
    if population is None:
        results_in_population = fitness.items()
    else:
        population = set(population)
        results_in_population = filter(lambda x: x[0] in population, fitness.items())
    return results_in_population

def best(fitness, population=None):
    results_in_population = _fitness_of_population(
        fitness, 
        population=population)
    return max(results_in_population, key=operator.itemgetter(1))[0]

def worst(fitness, population=None):
    results_in_population = _fitness_of_population(
        fitness, 
        population=population)
    return min(results_in_population, key=operator.itemgetter(1))[0]

def average(fitness, population=None):
    results_in_population = _fitness_of_population(
        fitness, 
        population=population)
    return numpy.mean([x[1] for x in results_in_population])

def best_ratio(fitness):
    best_fitness = fitness[best(fitness)]
    avg_fitness = average(fitness)
    if avg_fitness != 0.0:
        return (best_fitness - avg_fitness) / avg_fitness
    return 1.0

def best_ratio_convergence(
    iterations, 
    individuals, 
    genepool, 
    gene_properties, 
    fitness, 
    **kwargs):

    br = best_ratio(fitness)
    thresh = kwargs.get('best_ratio_thresh', 0.001)
    _logger.debug(
        'checking best ratio {0} with threshold {1}'.format(br, thresh))
    if br < thresh:
        previous_best_fitness = kwargs.get('previous_best_fitness', 0.0)
        best_individual = best(fitness)
        best_fitness = fitness[best_individual]
        if best_fitness >= previous_best_fitness:
            _logger.info(
                'converged due to best ratio of {0} with result {1}'.format(
                    br,
                    best_individual))
            return genepool[best_individual]
    return None

def max_iteration_convergence(
    iterations, 
    individuals, 
    genepool, 
    gene_properties, 
    fitness, 
    **kwargs):

    max_iterations = kwargs.get('max_iterations', 1000)
    _logger.debug(
        'checking iteration count {0} of {1}'.format(
            iterations, 
            max_iterations))
    if iterations >= max_iterations:
        result = best(fitness)
        _logger.info('converged due to max iteration {0} with result {1}'.format(
            iterations,
            result))
        return genepool[result]

def create_generation_callstack(callstack):
    def _on_generation(
        iterations, 
        individuals, 
        genepool, 
        gene_properties, 
        fitness, 
        **kwargs):

        result = None
        for func in callstack:
            if result is None:
                result = func(
                    iterations, 
                    individuals, 
                    genepool, 
                    gene_properties,
                    fitness, 
                    **kwargs)
        return result
    return _on_generation 

def search(population_size, gene_properties, get_fitness, on_generation, **kwargs):

    _logger.info('creating initial population')
    individuals, genepool = population.create(
        population_size, 
        gene_properties, 
        **kwargs)

    global_fitness = {}
    result = None
    iteration = 0 
    while result is None:
        fitness = {}

        _logger.info('measuring fitness of iteration {0}'.format(iteration))
        for individual, genes in genepool.items():
            if not (individual in global_fitness):
                fit = get_fitness(genes)
                fitness[individual] = fit
                global_fitness[individual] = fit
            else:
                fitness[individual] = global_fitness[individual]
            _logger.debug('fitness {0} for individual {1}'.format(fitness[individual], individual))

        result = on_generation(
            iteration, 
            individuals, 
            genepool, 
            gene_properties,
            fitness, 
            **kwargs)
        _logger.info('result of iteration {0} > {1}'.format(
            iteration,
            result))

        if result is None:
            _logger.debug('evolving next generation')
            individuals, genepool = population.evolve(
                individuals,
                genepool,
                gene_properties,
                fitness, 
                **kwargs)
            iteration += 1

    _logger.info('finished search with result {0}'.format(result))
    return result

