
import warnings
warnings.filterwarnings("ignore")

import sys
import glob
import time
import gc
import copy
import _pickle as pickle
from multiprocessing import Pool
from collections import defaultdict, Counter
from os.path import splitext, basename, join, isfile
from datetime import datetime

sys.path.append('/home/alexandrearaujo/library/faiss/')
import pandas as pd
import numpy as np
import faiss
from delf import feature_io


kaggle_folder = '/home/alexandrearaujo/kaggle/landmark-retrieval-challenge'
storage_folder = '/media/hdd1/kaggle/landmark-retrieval-challenge/'

data_folder = join(kaggle_folder, 'data')
img_folder = join(kaggle_folder, 'img')
sample_submission = join(kaggle_folder, 'data', 'sample_submission.csv')


def sanitize(x):
    """ convert array to a c-contiguous float array """
    return np.ascontiguousarray(x.astype('float32'))

def write_memmap_file(path, data, dtype='float32'):
    f = np.memmap(path, dtype=dtype, mode='w+', shape=data.shape)
    f[:] = data[:]

def read_memmap_file(path, dtype='float32'):
    return np.memmap(path, dtype=dtype, mode='r')

def pickle_load(path):
    """function to load pickle object"""
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')

def _pickle_dump(file, path):
    """function to dump picke object"""
    with open(path, 'wb') as f:
        pickle.dump(file, f, -1)

def _get_new_name(path):
    """rename file if file already exist"""
    i = 0
    new_path = path
    while isfile(new_path):
        ext = splitext(path)[1]
        new_path = path.replace(ext, '_{}{}'.format(i, ext))
        i += 1
    return new_path

def pickle_dump(file, path, force=False):
    """dump a file without deleting an existing one"""
    if force:
        _pickle_dump(file, path)
    elif not force:
        new_path = _get_new_name(path)
        _pickle_dump(file, new_path)

def map_index(array, index):
    ret = np.empty_like(array)
    for i, j in np.ndindex(array.shape):
        ix = array[i, j]
        ret[i, j] = index[ix] if ix != -1 else -1
    return ret

def sort_agg(values, query_ix):
    """sort and aggregate index from descriptors"""
    count = Counter(values)
    if query_ix in count.keys():
        count.pop(query_ix)
    count = list(count.items())
    sort = sorted(count, key=lambda x: x[1], reverse=True)
    return [i[0] for i in sort][:100]

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



def main():

    _STATUS_CHECK_ITERATIONS = 10000
    _BATCH_TRAIN = 25000000

    dim = 40 # dim of descriptors

    index_files = glob.glob(join(img_folder, 'feature_index_256x256', '*'))
    query_files = glob.glob(join(img_folder, 'feature_test_256x256', '*'))

    # debug
    # index_files = index_files[:20000]
    # query_files = query_files[:1000]

    # faiss
    nlist = 100
    m = 8
    quantizer = faiss.IndexFlatL2(dim)  # this remains the same
    # 8 specifies that each sub-vector is encoded as 8 bits
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
    is_train = False

    #######################
    #####    TRAIN    #####
    #######################
    
    num_images = len(index_files)
    xtrain = []
    filenames_index = []
    filenames_index_ix = []
    start = time.clock()
    total_descriptors = 0
    count = 0
    print('______ Loading Train ______ ', flush=True)
    for i, path in enumerate(index_files):
        filename = splitext(basename(path))[0]
        try:
            loc, _, desc, _, _ = feature_io.ReadFromFile(path)
        except Exception as error:
            print('problem with file {}. error : {}'.format(
                            filename, error), flush=True)
            desc = np.zeros((1, 40))

        # count the number of descriptors 
        total_descriptors += desc.shape[0]
        count += desc.shape[0]
        # record the index of the 
        filenames_index.append(filename)
        filenames_index_ix.extend([i] * desc.shape[0])
        # print status
        if i % _STATUS_CHECK_ITERATIONS == 0 and i != 0:
            elapsed = (time.clock() - start)
            print('Loading image {} out of {}, last {} images took {:.5f}'
                    ' seconds, total number of descriptors {}'.format(
                        i, num_images, _STATUS_CHECK_ITERATIONS, 
                        elapsed, total_descriptors), flush=True)
            start = time.clock()

        # accumulate
        if count < _BATCH_TRAIN:
            # fill the database
            xtrain.append(desc)

        # first training
        elif count >= _BATCH_TRAIN and not is_train:
            print('total number of descriptors reached, '
                    'adding descriptors to index', flush=True)
            xtrain = sanitize(np.concatenate(xtrain))
            index.train(xtrain)
            is_train = True
            # pickle_dump(xtrain, 'xtrain.pkl')
            # pickle_dump(filenames_index, 'filenames_index_train.pkl')
            count = 0
            xtrain = []

        # add index
        elif count >= _BATCH_TRAIN and is_train:
            print('total number of descriptors reached, '
                    'adding descriptors to index', flush=True)
            xtrain = sanitize(np.concatenate(xtrain))
            index.add(xtrain)
            count = 0
            xtrain = []

    pickle_dump(filenames_index, 'filenames_index.pkl')
    pickle_dump(filenames_index_ix, 'filenames_index_ix.pkl')


    ######################
    #####    TEST    #####
    ######################

    num_images = len(query_files)
    xtest = []
    filenames_query = []
    filenames_query_ix = []
    start = time.clock()
    print('______ Loading Test ______ ', flush=True)
    for i, path in enumerate(query_files):
        filename = splitext(basename(path))[0]
        try:
            loc, _, desc, _, _ = feature_io.ReadFromFile(path)
        except Exception as error:
            print('problem with file {}. error : {}'.format(
                            filename, error), flush=True)
            desc = np.zeros((1, 40))
        # count the number of descriptors 
        total_descriptors += desc.shape[0]
        # record the index of the 
        filenames_query.append(filename)
        filenames_query_ix.extend([i] * desc.shape[0])
        # print status
        if i % _STATUS_CHECK_ITERATIONS == 0 and i != 0:
            elapsed = (time.clock() - start)
            print('Loading image {} out of {}, last {} images took {:.5f}'
                    ' seconds, total number of descriptors {}'.format(
                        i, num_images, _STATUS_CHECK_ITERATIONS, 
                        elapsed, total_descriptors), flush=True)
            start = time.clock()
        # fill the database
        xtest.append(desc)
    xtest = sanitize(np.concatenate(xtest))
    pickle_dump(xtest, 'xtest.pkl')
    pickle_dump(filenames_query, 'filenames_query.pkl')
    pickle_dump(filenames_query_ix, 'filenames_query_ix.pkl')


    #########################
    #####    PREDICT    #####
    #########################

    print('______ Prediction ______ ', flush=True)

    k = 500
    ## Search on CPU
    # D, I = index.search(xtest, k)

    ## Search on GPU
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    D, I = gpu_index.search(xtest, k)

    print('save index & distances', flush=True)
    pickle_dump(D, 'distances.pkl')
    pickle_dump(I, 'index.pkl')

    print('mapped_index', flush=True)
    mapped_index = map_index(I, filenames_index_ix)

    print('map real index to img index', flush=True)
    # map real index to img index
    data = defaultdict(list)
    for i in filenames_query_ix:
        data[i].extend(mapped_index[i])
    
    print('sort and aggregate values form descriptors', flush=True)
    # sort and aggregate values form descriptors
    sorted_agg_data = {}
    for key, values in data.items():
        sorted_agg_data[key] = sort_agg(values)

    print('map image index to filename', flush=True)
    # map image index to filename
    results = {}
    for query_ix, query_result in sorted_agg_data.items():
        l = [filenames_index[ix] for ix in query_result if ix != -1]
        results[filenames_query[query_ix]] = ' '.join(l)

    print('make submit', flush=True)
    submit = pd.read_csv(sample_submission)
    submit['images'] = submit['id'].apply(lambda ix: results.get(ix, ''))

    date = datetime.now()
    submit_filename = 'submit_{:%Y-%m-%d}_full_index.csv.gz'.format(date)
    submit.to_csv(submit_filename, index=False, compression='gzip')

def run():

    # print('load index', flush=True)
    # # pickle_load('distances.pkl')
    # I = pickle_load('index.pkl')

    print('load filenames index', flush=True)
    # filenames_index_ix = pickle_load('filenames_index_ix.pkl')
    filenames_query_ix = pickle_load('filenames_query_ix.pkl')
    filenames_index = pickle_load('filenames_index.pkl')
    filenames_query = pickle_load('filenames_query.pkl')

    # print('mapped_index', flush=True)
    # # mapped_index = map_index(I, filenames_index_ix)
    # for i, j in np.ndindex(I.shape):
    #     ix = I[i, j]
    #     I[i, j] = filenames_index_ix[ix] if ix != -1 else -1
    # pickle_dump(I, 'mapped_index.pkl')

    print('load mapped_index', flush=True)
    # # I = pickle_load('mapped_index.pkl')
    # with open('mapped_index.pkl', 'rb') as f:
    #     I = pickle.load(f, encoding='latin1')
    #     I = I[:, :100].astype('uint32')
    # pickle_dump(I, 'mapped_index_compress.pkl')
    I = pickle_load('mapped_index_compress.pkl')

    nb_descriptors = len(I)
    _STATUS_CHECK_ITERATIONS = 100000
    # _NUM_THREADS = 12

    # I = np.array_split(I, _NUM_THREADS)
    # ix_split = np.array_split(np.arange(nb_descriptors), _NUM_THREADS)
    # # create the process pool
    # pool = Pool(processes=_NUM_THREADS)
    # # map the list of lines into a list of result dicts
    # I = merge_dicts(*pool.map(worker, zip(I, ix_split)))


    print('map real index to img index', flush=True)
    # map real index to img index
    start = time.clock()
    results = defaultdict(Counter)
    dump = False
    for i, ix in enumerate(filenames_query_ix):
        results[ix] += Counter(I[i])
        # print status
        if i % _STATUS_CHECK_ITERATIONS == 0 and i != 0:
            elapsed = (time.clock() - start)
            print('Aggregate descriptors {} out of {}, last {} descriptors'
                    ' took {:.5f} seconds'.format(i, nb_descriptors, 
                        _STATUS_CHECK_ITERATIONS, elapsed), flush=True)
            start = time.clock()
    I = None
    # pickle_dump(results, 'results_v1_part2.pkl')


    _STATUS_CHECK_ITERATIONS = 10000

    print('sort values form descriptors and map img index to filenames', flush=True)
    # sort and aggregate values form descriptors
    nb_query = len(results.keys())
    keys = list(results.keys())
    start = time.clock()
    for i, query_ix in enumerate(keys):
        values = results[query_ix]
        if query_ix in values.keys():
            values.pop(query_ix)
        values = sorted(values.items(), key=lambda x: x[1], reverse=True)[:100]
        # record result as string 
        results[filenames_query[query_ix]] = ' '.join([filenames_index[ix] for ix, _ in values if ix != -1])
        results.pop(query_ix)
        # print status
        if i % _STATUS_CHECK_ITERATIONS == 0 and i != 0:
            elapsed = (time.clock() - start)
            print('Sort and aggregate descriptors {} out of {}, last {} images'
                    ' took {:.5f} seconds'.format(i, nb_query, 
                        _STATUS_CHECK_ITERATIONS, elapsed), flush=True)
            start = time.clock()

    # pickle_dump(results, 'results.pkl')

    print('make submit', flush=True)
    submit = pd.read_csv(sample_submission)
    submit['images'] = submit['id'].apply(lambda ix: results.get(ix, ''))

    date = datetime.now()
    submit_filename = 'submit_{:%Y-%m-%d}_full_index.csv.gz'.format(date)
    submit.to_csv(submit_filename, index=False, compression='gzip')

    

def process_results(results):

    _STATUS_CHECK_ITERATIONS = 100000

    filenames_query_ix = pickle_load('filenames_query_ix.pkl')
    filenames_index = pickle_load('filenames_index.pkl')
    filenames_query = pickle_load('filenames_query.pkl')

    print('sort, aggregate values form descriptors and map img to filenames', flush=True)
    # sort and aggregate values form descriptors
    nb_query = len(results.keys())
    keys = results.items()
    start = time.clock()
    for i, query_ix in enumerate(keys):
        values = results[query_ix]
        if query_ix in values.keys():
            values.pop(query_ix)
        values = sorted(values.items(), key=lambda x: x[1], reverse=True)
        # record result as string 
        results[filenames_query[query_ix]] = ' '.join([filenames_index[ix] for ix in values if ix != -1])        
        results.pop(query_ix)
        # print status
        if i % _STATUS_CHECK_ITERATIONS == 0 and i != 0:
            elapsed = (time.clock() - start)
            print('Sort and aggregate descriptors {} out of {}, last {} images'
                    ' took {:.5f} seconds'.format(i, nb_query, 
                        _STATUS_CHECK_ITERATIONS, elapsed), flush=True)
            start = time.clock()
    return results


def make_submit():

    results_part1 = pickle_load('results_v1_part1.pkl')
    results_part1 = process_results(results_part1)
    pickle_dump(results_part1, 'results_v2_part1.pkl')

    results_part2 = pickle_load('results_v1_part2.pkl')
    results_part2 = process_results(results_part2)
    pickle_dump(results_part2, 'results_v2_part2.pkl')

    results = merge_dicts([results_part1, results_part2])
    results_part1 = None
    results_part2 = None

    # results = pickle_load('results_v1.pkl')
    # results = process_results(results)
    # pickle_dump(results, 'results_v2.pkl')


    print('make submit', flush=True)
    submit = pd.read_csv(sample_submission)
    submit['images'] = submit['id'].apply(lambda ix: results.get(ix, ''))

    date = datetime.now()
    submit_filename = 'submit_{:%Y-%m-%d}_full_index.csv.gz'.format(date)
    submit.to_csv(submit_filename, index=False, compression='gzip')


if __name__ == '__main__':
    # main()
    run()
    # make_submit()