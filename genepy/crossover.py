import numpy

def swap(a, b):
    return (b, a)

def sample_gaussian(a, b):
    mu = numpy.mean([a, b])
    sig = numpy.std([a, b])
    deviation = sig * numpy.random.randn()
    return (mu + deviation, mu - deviation)

