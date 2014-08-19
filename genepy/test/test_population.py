import pprint
import logging
import random

from genepy import population
from genepy import generate
from genepy import mutate
from genepy import crossover

_gene_properties = {
    'gene1': {},
    'gene2': {
        'generate': generate.constant(8.0),
    },
    'gene3': {
        'generate': generate.sample(random.randrange, 2, 12, 2),
        'mutate': mutate.sample(random.randrange, 2, 12, 2),
        'crossover': crossover.swap
    }
}

_kwargs = {
    'active_genes': ['gene1', 'gene3'],
    'count': 50
}

def test_create():
    pop, gene_pool = population.create(100, _gene_properties)

    assert(not pop is None)
    assert(not gene_pool is None)

    assert(len(pop) == 100)
    assert(len(gene_pool) <= 100)
    assert(len(gene_pool) > 0)

    for ind in pop:
        print gene_pool[ind]
        assert(ind in gene_pool)
        assert('gene1' in gene_pool[ind])
        assert('gene2' in gene_pool[ind])
        assert(gene_pool[ind]['gene1'] >= 0.0)
        assert(gene_pool[ind]['gene1'] <= 1.0)
        assert(gene_pool[ind]['gene2'] == 8.0)
        assert(gene_pool[ind]['gene3'] >= 2)
        assert(gene_pool[ind]['gene3'] <= 12)
        assert(gene_pool[ind]['gene3'] % 2 == 0)

def test_evolve():
    pop, gene_pool = population.create(100, _gene_properties)

    # give fitness as value of gene1
    fitness = dict(map(lambda x: (x, gene_pool[x]['gene1']), pop))

    pop2, gene_pool2 = population.evolve(
        pop, 
        gene_pool, 
        _gene_properties, 
        fitness, 
        **_kwargs)

    assert(not pop2 is None)
    assert(not gene_pool2 is None)
    assert(len(pop2) == 50)
    assert(len(gene_pool2) <= 50)
    assert(len(gene_pool2) > 0)

    # TODO: more tests on expected output

def test_select():
    pop, gene_pool = population.create(100, _gene_properties)

    # give fitness as position in pop
    fitness = dict(map(lambda x: (x[1], x[0]), enumerate(pop)))
    pprint.pprint(fitness)

    pop2, gene_pool2, best_genes = population.select(pop, gene_pool, fitness, **_kwargs)
    pprint.pprint(pop2)

    assert(not pop2 is None)
    assert(not gene_pool2 is None)
    assert(len(pop2) == 50)
    assert(len(gene_pool2) <= 50)
    assert(len(gene_pool2) > 0)
    assert(not best_genes is None)
    assert(best_genes == gene_pool[pop[-1]])

