
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
feature_folder = join(kaggle_folder, 'img')
sample_submission = join(kaggle_folder, 'data', 'sample_submission.csv')


def sanitize(x):
    """ convert array to a c-contiguous float array """
    return np.ascontiguousarray(x.astype('float32'))

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


def main():

    _STATUS_CHECK_ITERATIONS = 10000
    _BATCH_TRAIN = 25000000
    # _BATCH_TRAIN = 800000 # debug

    index_files = glob.glob(join(feature_folder, 'feature_index_256x256', '*'))
    query_files = glob.glob(join(feature_folder, 'feature_test_256x256', '*'))

    # # debug
    # index_files = index_files[:20000]
    # query_files = query_files[:1000]


    # # faiss
    # d = 40 # dim of descriptors
    # nlist = 2**5
    # m = 8 # number of subquantizers
    # quantizer = faiss.IndexFlatL2(d)  # this remains the same
    # n_bits = 8 # bits allocated per subquantizer
    # index = faiss.IndexIVFPQ(quantizer, d, nlist, m, n_bits)

    # # debug
    # d = 40 # dim of descriptors
    # nlist = 2**5
    # m = 8 # number of subquantizers
    # quantizer = faiss.IndexFlatL2(d)  # this remains the same
    # n_bits = 8 # bits allocated per subquantizer
    # index = faiss.IndexIVFPQ(quantizer, d, nlist, m, n_bits)

    # faiss
    d = 40 # dim of descriptors
    index = faiss.IndexFlatL2(d)  # this remains the same

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
        elif count >= _BATCH_TRAIN and index.is_trained == False:
            print('total number of descriptors reached, training and '
                    'adding descriptors to index', flush=True)
            xtrain = sanitize(np.concatenate(xtrain))
            index.train(xtrain)
            index.add(xtrain)
            count = 0
            xtrain = []

        # add index
        elif count >= _BATCH_TRAIN and index.is_trained == True:
            print('total number of descriptors reached, '
                    'adding descriptors to index', flush=True)
            xtrain = sanitize(np.concatenate(xtrain))
            index.add(xtrain)
            count = 0
            xtrain = []

    pickle_dump(filenames_index, 'filenames_index.pkl')
    pickle_dump(filenames_index_ix, 'filenames_index_ix.pkl')
    xtrain = None

    ######################
    #####    TEST    #####
    ######################

    num_images = len(query_files)
    xtest = []
    filenames_query = []
    filenames_query_ix = []
    filenames_query_ix_counter = []
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
        filenames_query_ix_counter.append((i, desc.shape[0]))
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
    ## Search on GPU
    k = 60
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    distances, index = gpu_index.search(xtest, k)
    xtest = None

    print('save index & distances', flush=True)
    pickle_dump(distances, 'distances.pkl')
    pickle_dump(index, 'index.pkl')

    # remove local descriptors with a distance superior to 0.8
    index[distances > 0.8] = -1
    distances = None

    # map local descriptors index to image index
    print('mapped_index', flush=True)
    mapped_index = np.empty_like(index)
    for i, j in np.ndindex(index.shape):
        ix = index[i, j]
        mapped_index[i, j] = filenames_index_ix[ix] if ix != -1 else -1
    pickle_dump(mapped_index, 'mapped_index.pkl')


    # extract image close to the query image
    print('map real index to img index', flush=True)
    _STATUS_CHECK_ITERATIONS = 10000
    nb_query = len(filenames_query_ix_counter)
    count_img_by_query = {}
    start_subset, end_subset = 0, 0
    start = time.clock()
    for i, (query_ix, n_descriptors) in enumerate(filenames_query_ix_counter):
        end_subset += n_descriptors
        value, count = np.unique(mapped_index[start_subset:end_subset], return_counts=True)
        count_img_by_query[query_ix] = Counter(dict(zip(value, count))).most_common(100)
        start_subset += n_descriptors
        # print status
        if i % _STATUS_CHECK_ITERATIONS == 0 and i != 0:
            elapsed = (time.clock() - start)
            print('Processed {} out of {} query image, last {} images'
                    ' took {:.5f} seconds'.format(i, nb_query, 
                        _STATUS_CHECK_ITERATIONS, elapsed), flush=True)
            start = time.clock()


    _STATUS_CHECK_ITERATIONS = 10000
    print('sort values form descriptors and map img index to filenames', flush=True)
    # sort and aggregate values form descriptors
    nb_query = len(count_img_by_query.keys())
    keys = list(count_img_by_query.keys())
    start = time.clock()
    for i, query_ix in enumerate(keys):
        values = count_img_by_query[query_ix]
        # record result as string
        count_img_by_query[filenames_query[query_ix]] = ' '.join([filenames_index[ix] for ix, _ in values if ix != -1])
        count_img_by_query.pop(query_ix)
        # print status
        if i % _STATUS_CHECK_ITERATIONS == 0 and i != 0:
            elapsed = (time.clock() - start)
            print('Sort and aggregate descriptors {} out of {}, last {} images'
                    ' took {:.5f} seconds'.format(i, nb_query, 
                        _STATUS_CHECK_ITERATIONS, elapsed), flush=True)
            start = time.clock()


    print('make submit', flush=True)
    submit = pd.read_csv(sample_submission)
    submit['images'] = submit['id'].apply(lambda ix: count_img_by_query.get(ix, ''))

    date = datetime.now()
    submit_filename = 'submit_{:%Y-%m-%d}_full_index.csv.gz'.format(date)
    submit.to_csv(submit_filename, index=False, compression='gzip')


if __name__ == '__main__':
    main()