
from multiprocessing import Pool
from collections import Counter, defaultdict
import numpy as np

_NUM_THREADS = 8


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = defaultdict(Counter)
    for dictionary in dict_args:
        for key, value in dictionary.items():
            result[key].update(value)
    return result

def worker(files):
    data, index = files
    results = {}
    for i, ix in enumerate(index):
        results[ix] = Counter(data[i])
    return results

I = np.random.randint(0, 1500000, size=(14674019, 250), dtype=np.uint32)
I = np.array_split(I, _NUM_THREADS)
ix_split = np.array_split(np.arange(14674019), _NUM_THREADS)

pool = Pool(processes=_NUM_THREADS)
result = pool.map(worker, zip(I, ix_split))