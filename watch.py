from time import *
import sys

class Watch(object):
    ALL_WATCHES = dict()
    WATCH_HEADER = "Watch"
    MAX_NAME_LEN = len(WATCH_HEADER)

    def __init__(self, name):
        if name in Watch.ALL_WATCHES:
            raise ValueError('A watch with the name "%s" already exists.' % name)

        Watch.ALL_WATCHES[name] = self
        self.name = name
        if len(name) > Watch.MAX_NAME_LEN:
            Watch.MAX_NAME_LEN = len(name)
        self.num_invocations = 0
        self.total_time = 0
        self.start_time = self.elapsed_time = None

    def start(self):
        self.check_stopped()
        self.num_invocations += 1
        self.start_time = time()

    def stop(self):
        try:
            self.elapsed_time = time() - self.start_time
            self.start_time = None
            self.total_time += self.elapsed_time
        except TypeError as e:
            print("Watch has not started", file=sys.err)
            raise e

    def get_time_components(self):
        comps = [0] * 4
        t = int(self.total_time)
        comps[3] = int((self.total_time - t) * 1000)
        comps[2] = t % 60
        t //= 60
        comps[1] = t % 60
        t //= 60
        comps[0] = t

        return comps

    def check_stopped(self):
        if self.start_time is not None:
            raise ValueError("Watch %s has not stopped." % self.name)
    
    @property
    def avg_time(self):
        if self.num_invocations == 0:
            return 0
        
        return self.total_time / self.num_invocations
    
    @classmethod
    def print_all(cls):
        dashes = '-' * 55;
        format_str = "{{0:{0}s}}| ".format(Watch.MAX_NAME_LEN + 2)
        print(dashes)
        print(format_str.format(Watch.WATCH_HEADER), end="")
        print(" Invocations  |     Time     | Average(s)")
        print(dashes)

        format_str = format_str + "   {1:7d}    | {2:02d}:{3:02d}:{4:02d}.{5:03d} |    {6:.2f}"
        for w in Watch.ALL_WATCHES.values():
            w.check_stopped()
            print(format_str.format(w.name, w.num_invocations,
                                    *w.get_time_components(), w.avg_time))
            print(dashes)
