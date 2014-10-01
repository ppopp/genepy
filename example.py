import genepy.ga
import genepy.generate
import genepy.mutate
import genepy.crossover

import math
import random
import logging

_logger = logging.getLogger(__name__)

_data = None
_slope = 0.0
_offset = 0.0
_noise = 0.0

def generate_data():
    '''generates 1000 data points along a line with noise'''
    global _data
    global _slope
    global _offset
    global _noise
    _slope = round(random.uniform(-100.0, 100.0), 2)
    _offset = round(random.uniform(-100.0, 100.0), 2)
    _noise = random.random()

    _data = []

    _logger.info('generate example data with slope {0}, offset {1}, and noise {2}'.format(_slope, _offset, _noise))
    for i in xrange(0, 1000):
        x = random.random() * 1000.0
        y = _slope * x + _offset + _noise * random.random()
        _data.append((x, y))
    return _data

def fitness(genes):
    global _data

    # create function that outputs estimated line value for x input
    line = lambda x: x * genes['slope'] + genes['offset']

    # measure mean squared error over all data
    mse = 0.0
    for x,y in _data:
        mse += math.pow(y - line(x), 2.0)
    mse /= len(_data)
    _logger.debug('MSE of {0} for slope {1} and offset {2}'.format(mse, genes['slope'], genes['offset']))

    # a better fitness should be a larger number, so take negative of mse as fitness
    return -1.0 * mse

def generation(
    iterations, 
    individuals, 
    genepool, 
    gene_properties, 
    fitness, 
    **kwargs):

    _logger.debug('finished iteration {0}'.format(iterations))
    # get best individual
    best = genepy.ga.best(fitness)
    # check if fitness is better than threshold
    if fitness[best] > -1.0:
        return best
    # return none if we haven't converged
    return None

def run_example():
    generate_data()

    num_individuals = 1000
    gene_properties = {
        'slope': {
            'generate': genepy.generate.sample(random.uniform, -100.0, 100.0),
            'mutate': genepy.mutate.gaussian(10.0),
            'crossover': genepy.crossover.sample_gaussian,
            'process': lambda x: round(x, 2)

        },
        'offset': {
            'generate': genepy.generate.sample(random.uniform, -100.0, 100.0),
            'mutate': genepy.mutate.gaussian(1.0),
            'crossover': genepy.crossover.sample_gaussian,
            'process': lambda x: round(x, 2)
        }
    }

    _logger.info('begin ga search with {0} individuals'.format(num_individuals))

    result = genepy.ga.search(
        num_individuals,
        gene_properties,
        fitness,
        generation,
        mutation_rate=0.3)

    _logger.info('found result {0}'.format(result))
    _logger.info('target slope {0} offset {1}'.format(_slope, _offset))



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_example()


