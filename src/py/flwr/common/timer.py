#### TEST ONLY ####
import timeit


class Timer:
    def __init__(self):
        self.named_time = {}

    def tic(self, name='default'):
        lst = self.named_time.setdefault(name, [])
        lst.append(-timeit.default_timer())

    def toc(self, name='default'):
        self.named_time[name][-1] += timeit.default_timer()

    def get(self, name):
        return sum(self.named_time[name])

    def get_all(self):
        return dict([(k, sum(v)) for k, v in self.named_time.items()])

