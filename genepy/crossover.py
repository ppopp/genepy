import numpy

def swap(a, b):
    return (b, a)

def sample_gaussian(a, b):
    mu = numpy.mean([a, b])
    sig = numpy.std([a, b])
    deviation = sig * numpy.random.randn()
    return (mu + deviation, mu - deviation)

def quantize(crossover_func, point):
    def _crossover(a, b):
        x, y = crossover_func(a, b)
        return (round(x, point), round(y, point))
    return _crossover
