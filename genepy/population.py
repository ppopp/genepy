'''
This module contains methods for evolving populations using both population 
and genetic operators.

The module primarily utilizes three data structures, a "population", "genepool"
and "gene_properties".

A "population" is a list of individuals in the population, where each 
individual is hashable. An individual is a key in the "genepool".  Individuals
may appear more than once in the population, which denotes that two or more
individuals in the population share the exact same genes.  By default, 
individuals are created by calling the str(..) method on a dictionary of genes.
This can be altered by setting the gene_hash_function.

A "genepool" is a dictionary between an individual in the population and their
corresponding genes.  The genes in a genepool are also dictionaries mapping
between an individuals gene name and gene value.  For example:
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
            'mutate': lambda x: return x * random.random(),
            'crossover': lambda x,y: return (y, x),
            'process': lambda x: round(x, 2)
        },
        'gene2': {
            ...
        }
        ...
    }
'''

import logging
import operator
import random

import numpy

# underscores on module imports to avoid name conflicts
import crossover as _crossover
import mutate as _mutate
import generate as _generate

_logger = logging.getLogger(__name__)


def _str_hash(o):
    """
    Makes a string from a dictionary, list, tuple or set to any level,     
    that should be hashable (elements of unordered types are sorted)
    """
    return _inner_str_hash(o).rstrip('"\'').lstrip('"\'')


def _inner_str_hash(o):
    if isinstance(o, tuple):
        return '(' + ', '.join(_inner_str_hash(e) for e in o) + ')'
    if isinstance(o, list):
        return '[' + ', '.join(_inner_str_hash(e) for e in o) + ']'
    elif isinstance(o, (set, frozenset)):
        return '{' + ', '.join(_inner_str_hash(e) for e in sorted(o)) + '}'
    elif isinstance(o, dict):
        return '{' + ', '.join('{0}: {1}'.format(_inner_str_hash(k),_inner_str_hash(v))
                                for k,v in sorted(o.items())) + '}'
    elif isinstance(o, basestring):
        return "'" + o + "'"
    else:
        return str(o)

gene_hash_function = _str_hash

class Error(Exception):
    pass

def _process_pass(val):
    return val

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
    Create a new population.

    Arguments:
        size            - Number of individuals in population.

        gene_properties - Gene propertes for creating genes.  See Module 
        documentation for more info on gene_properties.

    [optional]:
        active_genes   - Names of genes to vary. Genes in the gene pool, but 
        not in this list, will be left unchanged.
        [Default: all genes]
    
    Return:
        Tuple of population list and gene dictionary 
            (population, genepool) = create(...)
    '''
    active_genes = set(kwargs.get('active_genes', gene_properties.keys()))
    genepool = {}
    population = []
    for i in xrange(size):
        genes = {}
        for gene, parameters in gene_properties.items():
            if gene in active_genes:
                processor = parameters.get('process', _process_pass)
                generator = parameters.get('generate', _generate.sample(random.random))
                genes[gene] = processor(next(generator))
            else:
                genes[gene] = parameters
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
    Evolve existing population and genepool.
    
    Arguments:
        population      - The initial popopulation. See module level 
        documentation for more information.

        genepool        - The initial genepool. See module level documentation 
        for more information.

        gene_properties - The properties of genes which control mutation, and 
        crossover. See module level documentation for more information.

        fitness         - A dictionary mapping between individuals in the 
        population and their fitness scores. A gene with a higher greater 
        fitness score is considered more fit than one without.

    [optional]:
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
    '''
    _logger.info(
        'evolving {0} individuals with {1} different genes'.format(
            len(population), 
            len(genepool)))

    # select
    population, genepool, best_genes = select(
        population, 
        genepool, 
        fitness, 
        **kwargs)
    _logger.debug('population size {0} after select'.format(len(population)))
    _logger.debug('genepool size {0} after select'.format(len(genepool)))

    # mutation
    population, genepool = mutate(
        population, 
        genepool, 
        gene_properties,
        **kwargs)
    _logger.debug('population size {0} after mutate'.format(len(population)))
    _logger.debug('genepool size {0} after mutate'.format(len(genepool)))

    # crossover
    population, genepool = crossover(
        population, 
        genepool, 
        gene_properties,
        **kwargs)
    _logger.debug('population size {0} after crossover'.format(len(population)))
    _logger.debug('genepool size {0} after crossover'.format(len(genepool)))

    # add copies of best genes from previous generation
    population, genepool = replace(
        population, 
        genepool, 
        best_genes, 
        **kwargs)
    _logger.debug('population size {0} after replace'.format(len(population)))
    _logger.debug('genepool size {0} after replace'.format(len(genepool)))

    return population, genepool

def select(
    population, 
    genepool, 
    fitness, 
    **kwargs):
    '''
    Select a new population from existing population using tournament style 
    selection.

    Arguments:
        population      - The initial popopulation. See module level 
        documentation for more information.

        genepool        - The initial genepool. See module level documentation 
        for more information.

        fitness         - A dictionary mapping between individuals in the 
        population and their fitness scores. A gene with a higher greater 
        fitness score is considered more fit than one without.

    [optional]:
        count          - Population size. Number of individuals to make it to 
        next generation.
        [Default: original population size]
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
    new_genepool = {}
    for k in set(new_population):
        new_genepool[k] = genepool[k]


    return new_population, new_genepool, best_genes

def mutate(
    population, 
    genepool, 
    gene_properties,
    **kwargs):
    '''
    Randomly mutate individual genes in population. 

    Arguments:
        population      - The initial popopulation. See module level 
        documentation for more information.

        genepool        - The initial genepool. See module level documentation 
        for more information.

        gene_properties - The properties of genes which control mutation, and 
        crossover. See module level documentation for more information.

    [optional]:
        active_genes   - Names of genes to vary. Genes in the gene pool, but 
        not in this list, will be left unchanged.
        [Default: all genes]

        mutation_rate  - Fraction of individuals in new generation which have 
        one param modified via mutation.
        [Default: 1.0 / (len(population) * sqrt(len(active_genes))]
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
            processor = gene_properties[g].get('process', _process_pass)
            mutator = gene_properties[g].get(
                'mutate', 
                _mutate.sample(random.random))
            old_val = new_genes[g]
            new_genes[g] = processor(mutator(old_val))
            _logger.debug(
                'mutate: %u, gene: %s, %g -> %g', 
                i, 
                g, 
                old_val, 
                new_genes[g])

        individual = gene_hash_function(new_genes)
        new_genepool[individual] = new_genes
        new_population.append(individual)

    _logger.info('mutation rate: {0}'.format(mutation_rate))
    _logger.info('num mutations: {0} ({1:.4f})'.format(
        num_mutations, 
        float(num_mutations) / (size * num_genes)))

    return new_population, new_genepool

def cross(genes1, genes2, gene_properties, **kwargs):
    '''
    Mate two sets of genes using uniform crossover.

    Arguments:
        genes1          - Set of parent genes.

        genes2          - Set of parent genes.

        gene_properties - The properties of genes which control mutation, and 
        crossover. See module level documentation for more information.

    [optional]:
        active_genes   - Names of genes to vary. Genes in the gene pool, but 
        not in this list, will be left unchanged.
        [Default: all genes]

        mixing_ratio   - Ratio of gene mixing when crossover is performed 
        between parents.
        [Default: 0.5]
    '''

    mixing_ratio = kwargs.get('mixing_ratio', 0.5)
    active_genes = kwargs.get(
        'active_genes', 
        gene_properties.keys())

    new_genes1 = dict(genes1)
    new_genes2 = dict(genes2)

    for gene in active_genes:
        if numpy.random.binomial(1, mixing_ratio) == 1:
            processor = gene_properties[gene].get('process', _process_pass)
            crosser = gene_properties[gene].get('crossover', _crossover.swap)
            g1, g2 = crosser(genes1[gene], genes2[gene])
            new_genes1[gene] = processor(g1)
            new_genes2[gene] = processor(g2)

    return new_genes1, new_genes2

def crossover(
    population, 
    genepool, 
    gene_properties,
    **kwargs):
    '''
    Randomly choose and mate individuals in population.  Replace them with 
    their offspring.

    Arguments:
        population      - The initial popopulation. See module level 
        documentation for more information.

        genepool        - The initial genepool. See module level documentation 
        for more information.

        gene_properties - The properties of genes which control mutation, and 
        crossover. See module level documentation for more information.

    [optional]:
        active_genes   - Names of genes to vary. Genes in the gene pool, but 
        not in this list, will be left unchanged.
        [Default: all genes]

        crossover_rate - Fraction of individuals in new generation which serve 
        as parents to new individuals.
        [Default: 0.5]

        mixing_ratio   - Ratio of gene mixing when crossover is performed 
        between parents.
        [Default: 0.5]
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

    _logger.info('set crossover rate: {0}'.format(crossover_rate))
    _logger.info('actual crossover rate: {0} ({1:.4f})'.format(
        len(maters), float(len(maters))/size))

    return new_population, new_genepool

def replace(
    population, 
    genepool, 
    best_genes,
    **kwargs):
    '''
    Randomly replace some individuals with the best individuals from previous 
    generation.

    Arguments:
        population      - The initial popopulation. See module level 
        documentation for more information.

        genepool        - The initial genepool. See module level documentation 
        for more information.

        best_genes      - Genes with highest fitness score

    [optional]:
        num_replaces   - Number of individuals to replace with copies of the 
        best config from previous generation.
        [Default: 1]
    '''
    num_replaces = kwargs.get('num_replaces', 1)

    if num_replaces < 1:
        return population, genepool

    _logger.info('replace') 

    size = len(population)
    new_population = list(population)
    best_individual = gene_hash_function(best_genes)
    for i in numpy.random.randint(0, size, size=num_replaces):
        _logger.info('replace ind: {0}'.format(i))
        new_population[i] = best_individual

    new_genepool = dict(genepool)
    new_genepool[best_individual] = best_genes

    # remove replaced keys from new dict
    new_population_set = set(new_population)
    new_genepool2 = {}
    for k, v in new_genepool.iteritems():
        if k in new_population_set:
            new_genepool2[k] = v

    return new_population, new_genepool2

