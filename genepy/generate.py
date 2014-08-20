
def sample(func, *args):
    while True:
        yield func(*args)

def constant(val):
    while True:
        yield val

def quantize(generator, point):
    while True:
        val = next(generator)
        yield round(val, point)

