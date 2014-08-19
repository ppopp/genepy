
def sample(func, *args):
    while True:
        yield func(*args)

def constant(val):
    while True:
        yield val
