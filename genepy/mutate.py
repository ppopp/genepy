import numpy

def gaussian(std):
    def _mutate(val):
        return val + std * numpy.random.randn()
    return _mutate

def population_gaussian(population, genepool, gene):
    mean = numpy.mean([genepool[k][gene] for k in population])
    return gaussian(mean * 0.3)

def sample(func, *args):
    def _mutate(val):
        return func(*args)
    return _mutate

