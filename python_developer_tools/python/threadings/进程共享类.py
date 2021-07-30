import multiprocessing
from multiprocessing.managers import BaseManager, NamespaceProxy


class Counter(object):
    def __init__(self):
        self.value = 0

    def update(self, value):
        self.value += value


def update(counter_proxy, thread_id):
    counter_proxy.update(1)


class CounterManager(BaseManager):
    pass


class CounterProxy(NamespaceProxy):
    _exposed_ = ('__getattribute__', '__setattr__', '__delattr__', 'update')

    def update(self, value):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod(self.update.__name__, (value,))


CounterManager.register('Counter', Counter, CounterProxy)


def main():
    manager = CounterManager()
    manager.start()

    counter = manager.Counter()
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for i in range(10):
        pool.apply(func=update, args=(counter, i))
    pool.close()
    pool.join()

    print('Should be 10 but is %s.' % counter.value)


if __name__ == '__main__':
    main()
