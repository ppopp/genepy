'''
This module contains methods for performing heuristic searches using a genetic 
algorithm. It exposes control over how gene's are mutated and crossed, as well 
as how the next generation population is selected.  Additionally, it offers two 
callbacks, one to calculate the fitness of a set of genes, and a second 
triggered after each generation to allow for stoping conditions, logging, and 
other functionality.

Setting up and controlling the algorithm is performed through 3 basic 
mechanisms, the "gene_properties", the "get_fitness" callback and the 
"on_generation" callback.

The "on_generation" callback is called after each new generation is created and 
their fitness calculated. If the "on_generation" callback returns anything 
other than None, the search is halted. It has the following signature:

    def on_generation(
        iterations, 
        individuals, 
        genepool, 
        gene_properties, 
        fitness, 
        **kwargs):

    iterations      - The number of generations thus far.

    individuals     - A list of individuals in the population. By default, 
    individuals are created by calling the str(..) method on a dictionary of 
    genes. This can be altered by setting the gene_hash_function in the 
    genepy.population module.

    genepool        - A dictionary between an individual in the population and 
    their corresponding genes.  The genes in a genepool are also dictionaries 
    mapping between an individuals gene name and gene value.  
    
    For example:
    {
        'individual1': {
            'gene1': 0.5,
            'gene2': 0.3
        },
        'individual2': {
            'gene1': 0.8,
            'gene2': 0.1
        },
    }

    gene_properties - The properties used to evelove the population. Further 
    documentation is available in the genepy.population module.

    fitness         - A dictionary mapping individuals to their fitness scores.

The "get_fitness" callback is called with the genes of an individual.  It's 
important to note that the fitness results of identical genes are cached for 
the entirety of the search. The "get_fitness" callback must accept a single 
argument which is the genes for an individual, and return a fitness score. 
Higher scores are interpreted as denoting that the individual is more fit.  

    def fitness(genes):
        ...
        calculate fitness score
        ...
        return score

    genes - A dictionary containing gene values.

If "batch" is set to "True", the fitness function should accept a dictionary
of individuals as keys and genes as values and return a dictionary with 
individuals as keys and fitnesses as values.

The "gene_properties" provide functions to generated, mutate, and crossover 
gene values.  In addition, they allow a general post processing function that
is applied any time a new gene value is created.  This can be useful if you
want to things such as quantize the values or ensure they are within a
particular range.  The fields of gene_properties are all optional, with default 
functionality useful for gene values which generally live in the range 
(0.0, 1.0).

"gene_properties" support the following fields:
    "generate"  - A python generator which yields initial values for genes

    "mutate"    - A function which accepts the previous gene value and produces
    a new gene value.

    "crossover" - A function which accepts two gene values and returns a tuple
    containing two new values.  The order of the individuals which produced the
    gene values is maintained during assignment. If you wanted to swap the gene
    values between the two individuals, you would provide a function such as:
    def swap(a, b):
        return (b, a)

    "process"   - A function which accepts a single argument and returns a 
    single argument.  The "process" function is applied whenever a new gene
    value is created through the "generate", "mutate", or "crossover" 
    functions.

    Example "gene_properties":

    def uniform_generator(min, max):
        while True:
            yield random.uniform(min, max)

    {
        'gene1': {
            'generate': uniform_generator(-5.0, 5.0),
            'mutate': lambda x: x * random.random(),
            'crossover': lambda x,y: (y, x),
            'process': lambda x: round(x, 2)
        },
        'gene2': {
            ...
        }
        ...
    }
'''
import operator
import logging
import multiprocessing

import numpy

import population

_logger = logging.getLogger(__name__)

class Error(Exception):
    pass

def _fitness_of_population(fitness, population=None):
    results_in_population = None
    if population is None:
        results_in_population = fitness.items()
    else:
        population = set(population)
        results_in_population = filter(lambda x: x[0] in population, fitness.items())
    return results_in_population

def best(fitness, population=None):
    '''Return most fit individual in population.'''
    results_in_population = _fitness_of_population(
        fitness, 
        population=population)
    return max(results_in_population, key=operator.itemgetter(1))[0]

def worst(fitness, population=None):
    '''Return least fit individual in population.'''
    results_in_population = _fitness_of_population(
        fitness, 
        population=population)
    return min(results_in_population, key=operator.itemgetter(1))[0]

def average(fitness, population=None):
    '''Average fitness of population'''
    results_in_population = _fitness_of_population(
        fitness, 
        population=population)
    return numpy.mean([x[1] for x in results_in_population])

def best_ratio(fitness):
    '''
    Ratio of difference between best fitness and average fitness.

    best_ratio = (best_fitness - avg_fitness) / avg_fitness
    '''
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
    '''
    Tests for convergence of best_ratio below a threshold.

    [optional]
    best_ratio_thresh - The upper threshold for convergence.
    [Default: 0.001]
    '''

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
    '''
    Tests for convergence of maximum iterations.'

    [optional]
    max_iterations - The maximum number of iterations
    [Default: 1000]
    '''

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
    '''
    Creates a function which calls all the on_generation functions supplied in
    the callstack. If a function returns a value other than None, the 
    subsequent functions will not be called.

    callstack - A list of functions with the on_generation signature.

        def on_generation(
            iterations, 
            individuals, 
            genepool, 
            gene_properties, 
            fitness, 
            **kwargs)
    '''

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

def _get_fitness(get_fitness, genepool, global_fitness):
    # get fitness using current process
    fitness = {}
    for individual, genes in genepool.items():
        if not (individual in global_fitness):
            fit = get_fitness(genes)
            fitness[individual] = fit
            global_fitness[individual] = fit
        else:
            fitness[individual] = global_fitness[individual]
    return fitness

def _get_batch_fitness(get_fitness, genepool, global_fitness):
    # get in batches
    fitness = {}
    to_run = {}
    for individual, genes in genepool.items():
        if not (individual in global_fitness):
            to_run[individual] = genes
        else:
            fitness[individual] = global_fitness[individual]

    new_fitness = get_fitness(to_run)
    fitness.update(new_fitness)
    global_fitness.update(new_fitness)

    return fitness

def _update_fitness_multiprocess(
    get_fitness, 
    genepool, 
    global_fitness, 
    processor_count, 
    timeout, 
    timeout_fitness):

    # get fitness using a multiprocessor pool
    fitness = {}
    pool = None
    try:
        pool = multiprocessing.Pool(processor_count)
        async_fitness = {}
        for individual, genes in genepool.items():
            if not (individual in global_fitness):
                async_fitness[individual] = pool.apply_async(get_fitness, [genes])
            else:
                fitness[individual] = global_fitness[individual]
        pool.close()
        for individual, async_result in async_fitness.items():
            try:
                fitness[individual] = async_result.get(timeout=timeout)
            except multiprocessing.TimeoutError, e:
                _logger.warning(
                    'fitness function timeout after {0} seconds with genes {1}'.format(
                        timeout,
                        genepool[individual]))
                if not (timeout_fitness) is None:
                    fitness[individual] = timeout_fitness
                else:
                    raise
    except:
        if pool:
            pool.terminate()
        raise
    finally:
        if pool:
            pool.join()
    return fitness


def search(population_size, gene_properties, get_fitness, on_generation, **kwargs):
    '''
    [optional]:
        individuals    - Prexisting set of individuals to begin with.
        [Default: None]

        genepool       - Prexistings set of genes associated with genepool.
        [Default: None]

        fitness        - Prexisting fitness results of individuals.
        [Default: None]

        active_genes   - Names of genes to vary. Genes in the gene pool, but 
        not in this list, will be left unchanged.
        [Default: all genes]

        count          - Population size. Number of individuals to make it to 
        next generation.
        [Default: original population size]

        crossover_rate - Fraction of individuals in new generation which serve 
        as parents to new individuals.
        [Default: 0.5]

        mixing_ratio   - Ratio of gene mixing when crossover is performed 
        between parents.
        [Default: 0.5]

        mutation_rate  - Fraction of individuals in new generation which have 
        one param modified via mutation.
        [Default: 1.0 / (len(population) * sqrt(len(active_genes))]

        num_replaces   - Number of individuals to replace with copies of the 
        best config from previous generation.
        [Default: 1]

        processor_count
        [Default: 1]

        timeout
        [Default: None]

        timeout_fitness
        [Default: None]
    '''

    # get optional parameters
    processor_count = kwargs.get('processor_count', 1)
    _logger.debug('setting processor count to {0}'.format(processor_count))
    timeout = kwargs.get('timeout', None)
    _logger.debug('setting timeout to {0}'.format(timeout))
    timeout_fitness = kwargs.get('timeout_fitness', None)
    if (timeout is None) and processor_count > 1:
        _logger.warning('timeout not applied when using single process')

    do_batch = kwargs.get('batch', False)
    _logger.debug('setting batch mode to {0}'.format(do_batch))

    if (processor_count > 1) and do_batch:
        _logger.warning('batch mode only allows single process.  ignoring processor_count')

    # get existing population and results
    individuals = kwargs.get('individuals')
    genepool = kwargs.get('genepool')
    fitness = kwargs.get('fitness')

    # need to remove these from kwargs so further function calls don't get 
    # duplicate values for same keyword
    if individuals:
        del kwargs['individuals']
    if genepool:
        del kwargs['genepool']
    if fitness:
        del kwargs['fitness']

    # create initial generation
    if (individuals is None) and (genepool is None):
        _logger.info('creating initial population')
        individuals, genepool = population.create(
            population_size, 
            gene_properties, 
            **kwargs)
    elif (individuals is None) or (genepool is None):
        raise Error({
            'message': 'must supply both individuals and genepool or neither as optional parameters',
            'supplied_inidividuals': not (individuals is None),
            'supplied_genepool': not (genepool is None)
        })



    # search until result is found
    global_fitness = {}
    if fitness:
        global_fitness.update(fitness)
    result = None
    iteration = 0 
    while result is None:
        fitness = None
        tests = []

        _logger.info('measuring fitness of iteration {0}'.format(iteration))
        if do_batch:
            fitness = _get_batch_fitness(get_fitness, genepool, global_fitness)
        if processor_count > 1:
            fitness = _update_fitness_multiprocess(
                get_fitness, 
                genepool, 
                global_fitness, 
                processor_count, 
                timeout, 
                timeout_fitness)
        else:
            fitness = _get_fitness(get_fitness, genepool, global_fitness)

        for individual in individuals:
            _logger.debug('fitness {0} for individual {1}'.format(
                fitness[individual], 
                individual))

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

