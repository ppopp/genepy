import logging
import operator
import random

import numpy

# underscores on module imports to avoid name conflicts
import crossover as _crossover
import mutate as _mutate
import generate as _generate

_logger = logging.getLogger(__name__)


gene_hash_function = str

class Error(Exception):
    pass

def _sample_with_replace(population, k):
    "Chooses k random elements (with replace) from a population"
    n = len(population)
    _random, _int = random.random, int  # speed hack 
    result = [None] * k
    for i in xrange(k):
        j = _int(_random() * n)
        result[i] = population[j]
    return result

def create(size, gene_properties, **kwargs):
    '''
    Create a random population.

    Arguments:
        size - Number of individuals in population.
        gene_properties - Dictionary of gene names and parameters
            gene_properties = {
                'gene1': {
                    'generate': generate.sample(random.uniform, -10, 10)
                    'crossover': crossover.swap,
                    'mutate': mutate.gaussian(0.5)
                },
                'gene2': {
                    'generator': generate.constant(7.3)
                }
            }
    
    Return:
        Tuple of population list and gene dictionary 
            (population, genepool) = create(...)
    '''
    genepool = {}
    population = []
    for i in xrange(size):
        genes = {}
        for gene, parameters in gene_properties.items():
            generator = parameters.get('generate', _generate.sample(random.random))
            genes[gene] = next(generator)
        individual = gene_hash_function(genes)
        genepool[individual] = genes
        population.append(individual)
    return population, genepool

def evolve(
        population,
        genepool,
        gene_properties,
        fitness,
        **kwargs):
    '''
    create a new generation of parameters
    population - 
    genepool - 
    fitness - 
    gene_properties - TODO

    kwargs:
    active_genes - names of parameters to vary
    count - population size, number of individuals to make it to next generation
    crossover_rate - percentage of configs in new generation
                     which serve as parents to new configs
    mutation_rate - percentage of configs in new generation
                    which have one param modified via mutation
    num_replaces - number of configs to replace with copies
                       of the best config from previous generation
    mixing_ratio - Ratio of gene mixing when crossover is performed between parents

    Notes:
        Assuming real-valued parameters.
        Population size - 25--100
        Crossover rate - high for binary genes/genepool (~95%),
                         lower for real genes/genepool (~10%)
        Mutation rate - 1./(M*L**0.5), M is size,
                        L is number of genes/genepool per config
    '''
    _logger.info('computing new generation')

    # select
    population, genepool, best_genes = select(
        population, 
        genepool, 
        fitness, 
        **kwargs)

    # mutation
    population, genepool = mutate(
        population, 
        genepool, 
        gene_properties,
        **kwargs)

    # crossover
    population, genepool = crossover(
        population, 
        genepool, 
        gene_properties,
        **kwargs)

    # add copies of best genes from previous generation
    population, genepool = replace(
        population, 
        genepool, 
        best_genes, 
        **kwargs)

    return population, genepool

def select(
    population, 
    genepool, 
    fitness, 
    **kwargs):
    '''
    select new population from existing population
    '''
    _logger.info('selection')

    count = kwargs.get('count', len(population))
    # tournament-style selection
    best_genes = genepool[
        max
        (fitness.iteritems(), key=operator.itemgetter(1))
        [0]]
    # to support older versions of numpy pre 1.7.0
    team1 = _sample_with_replace(population, count)
    team2 = _sample_with_replace(population, count)
    new_population = [
        t1 if fitness[t1] >= fitness[t2] else t2 for t1,
        t2 in zip(
            team1,
            team2)]
    new_genepool = {k: genepool[k] for k in set(new_population)}

    return new_population, new_genepool, best_genes

def mutate(
    population, 
    genepool, 
    gene_properties,
    **kwargs):

    '''
    add random gaussian noise to each mutated gene
    '''

    size = len(population)
    active_genes = kwargs.get(
        'active_genes', 
        gene_properties.keys())
    num_genes = len(active_genes)
    mutation_rate = kwargs.get(
        'mutation_rate',
        1. / (size * numpy.sqrt(num_genes)))

    new_population = []
    new_genepool = {}
    num_mutations = 0

    _logger.info('mutating')
    for i, k in enumerate(population):
        new_genes = dict(genepool[k])
        mutate_genepool = [
            g
            for g in
            active_genes
            if numpy.random.binomial(1, mutation_rate) == 1]
        num_mutations += len(mutate_genepool)
        for g in mutate_genepool:
            mutator = gene_properties[g].get(
                'mutate', 
                _mutate.sample(random.random))
            old_val = new_genes[g]
            new_genes[g] = mutator(old_val)
            _logger.debug(
                'mutate: %u, gene: %s, %g -> %g', 
                i, 
                g, 
                old_val, 
                new_genes[g])

        individual = gene_hash_function(new_genes)
        new_genepool[individual] = new_genes
        new_population.append(individual)

    _logger.info('mutation rate: {}'.format(mutation_rate))
    _logger.info('num mutations: {} ({:.4f})'.format(
        num_mutations, 
        float(num_mutations) / (size * num_genes)))

    return new_population, new_genepool

def cross(genes1, genes2, gene_properties, **kwargs):
    '''
    mate two sets of genes using uniform crossover
    '''

    mixing_ratio = kwargs.get('mixing_ratio', 0.5)
    active_genes = kwargs.get(
        'active_genes', 
        gene_properties.keys())

    new_genes1 = dict(genes1)
    new_genes2 = dict(genes2)

    for gene in active_genes:
        if numpy.random.binomial(1, mixing_ratio) == 1:
            crosser = gene_properties[gene].get('crossover', _crossover.swap)
            g1, g2 = crosser(genes1[gene], genes2[gene])
            new_genes1[gene] = g1
            new_genes2[gene] = g2

    return new_genes1, new_genes2

def crossover(
    population, 
    genepool, 
    gene_properties,
    **kwargs):
    '''
    crossover_rate - percentage of configs in new generation
                     which serve as parents to new configs
    '''

    size = len(population)
    crossover_rate = kwargs.get('crossover_rate', 0.5)

    new_genepool = {}
    new_population = []

    _logger.info('crossover') 

    # choose which to mate and which to keep single
    maters = []
    for ind in population:
        if numpy.random.binomial(1, crossover_rate):
            maters.append(ind)
        else:
            new_population.append(ind)

    # handle odd number of maters
    if (len(maters) % 2) != 0:
        if numpy.random.binomial(1, 0.5) or (len(new_population) == 0):
            # randomly remove one mater and add to new_population
            pos = random.randint(0, len(maters) - 1)
            new_population.append(maters[pos])
            del maters[pos]
        else:
            # randomly remove on single and add to maters
            pos = random.randint(0, len(new_population) - 1)
            maters.append(new_population[pos])
            del new_population[pos]

    # add new_population to new_genepool
    for ind in new_population:
        new_genepool[ind] = dict(genepool[ind])

    # mate and add children
    random.shuffle(maters)
    for i in xrange(0, len(maters), 2):
        k1 = maters[i]
        k2 = maters[i + 1]
        genes1, genes2 = cross(
            genepool[k1], 
            genepool[k2], 
            gene_properties, 
            **kwargs)
        ind1 = gene_hash_function(genes1)
        ind2 = gene_hash_function(genes2)
        new_genepool[ind1] = genes1
        new_genepool[ind2] = genes2
        new_population.append(ind1)
        new_population.append(ind2)
        _logger.debug('parents:\n\t{0}\n\t{1}'.format(k1, k2))
        _logger.debug('children:\n\t{0}\n\t{1}'.format(ind1, ind2))

    _logger.info('set crossover rate: {}'.format(crossover_rate))
    _logger.info('actual crossover rate: {} ({:.4f})'.format(
        len(maters), float(len(maters))/size))

    return new_population, new_genepool

def replace(
    population, 
    genepool, 
    best_genes,
    **kwargs):
    '''
    randomly replace some configs with the best genes from previous generation
    '''
    num_replaces = kwargs.get('num_replaces', 1)

    if num_replaces < 1:
        return population, genepool

    _logger.info('replace') 

    size = len(population)
    new_population = list(population)
    best_individual = gene_hash_function(best_genes)
    for i in numpy.random.randint(0, size, size=num_replaces):
        _logger.info('replace ind: {}'.format(i))
        new_population[i] = best_individual

    new_genepool = dict(genepool)
    new_genepool[best_individual] = best_genes

    # remove replaced keys from new dict
    new_genepool = {
        k: v for k,
        v in new_genepool.iteritems() if k in new_population}

    return new_population, new_genepool

