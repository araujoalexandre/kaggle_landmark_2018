
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

sys.path.append('/home/alexandrearaujo/libraries/faiss/')
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

    index_files = glob.glob(join(storage_folder, 'feature_index_rescale_delf', '*'))
    query_files = glob.glob(join(storage_folder, 'feature_test_rescale_delf', '*'))

    
    # #######################
    # #####    DEBUG    #####
    # #######################
    # index_files = index_files[:5000]
    # query_files = query_files[:1000]
    # _BATCH_TRAIN = 2000000
    # _STATUS_CHECK_ITERATIONS = 1000

    # kmeans
    dim = 40 # dim of descriptors
    ncentroids = 16
    niter = 100
    verbose = True
    kmeans = faiss.Kmeans(dim, ncentroids, niter, verbose)

    # index
    index = faiss.IndexFlatL2(ncentroids * dim)

    #######################
    #####    TRAIN    #####
    #######################
    
    num_images = len(index_files)
    xtrain = []
    filenames_index = []
    filenames_index_ix = []
    ndesc_by_img = []
    start = time.clock()
    total_descriptors = 0
    ndesc_in_batch = 0
    nimg = 0
    is_train = False
    print('______ Loading Train ______ ', flush=True)
    for i, path in enumerate(index_files):
        filename = splitext(basename(path))[0]
        
        try:
            loc, _, desc, _, _ = feature_io.ReadFromFile(path)
        except Exception as error:
            print('problem with file {}. error : {}'.format(
                            filename, error), flush=True)
            desc = np.zeros((1, 40))

        # ndesc_in_batch the number of descriptors 
        ndesc_in_batch += desc.shape[0]
        
        # record the index of the 
        filenames_index.append(filename)
        filenames_index_ix.append(desc.shape[0])

        # batch variables 
        ndesc_by_img.append(desc.shape[0])
        nimg += 1

        # print status
        total_descriptors += desc.shape[0]
        if i % _STATUS_CHECK_ITERATIONS == 0 and i != 0:
            elapsed = (time.clock() - start)
            print('Loading image {} out of {}, last {} images took {:.5f}'
                    ' seconds, total number of descriptors {}'.format(
                        i, num_images, _STATUS_CHECK_ITERATIONS, 
                        elapsed, total_descriptors), flush=True)
            start = time.clock()

        # accumulate
        if ndesc_in_batch < _BATCH_TRAIN:
            # fill the database
            xtrain.append(desc)

        # first training
        elif ndesc_in_batch >= _BATCH_TRAIN and is_train == False:
            print('total number of descriptors reached, training kmeans '
                    'and adding descriptors to index', flush=True)
            
            xtrain = sanitize(np.concatenate(xtrain))
            kmeans.train(xtrain)
            is_train = True
            _, I = kmeans.index.search(xtrain, 1)
            _ = None
            I = I.flatten()

            nimg = len(ndesc_by_img)
            # vlad
            xtrain = xtrain - kmeans.centroids[I]
            database = np.empty((nimg, ncentroids, dim))
            for cluster in range(ncentroids):
                start_ix = 0
                for ix, end_ix in enumerate(ndesc_by_img):
                    mask = I[start_ix:end_ix] == cluster
                    database[ix, cluster, :] = \
                        xtrain[start_ix:end_ix][mask, :].sum(axis=0)
                    start_ix = end_ix
            database = database.reshape(nimg, ncentroids * dim)
            database = sanitize(database)

            # first train the index
            index.train(database)
            index.add(database)
            
            # reset
            ndesc_in_batch = 0
            xtrain, database, ndesc_by_img = [], [], []

        # add index
        elif ndesc_in_batch >= _BATCH_TRAIN and is_train == True:
            print('total number of descriptors reached, '
                    'adding descriptors to index', flush=True)
            
            xtrain = sanitize(np.concatenate(xtrain))
            _, I = kmeans.index.search(xtrain, 1)
            _ = None
            I = I.flatten()

            nimg = len(ndesc_by_img)

            # vlad
            xtrain = xtrain - kmeans.centroids[I]
            database = np.empty((nimg, ncentroids, dim))
            for cluster in range(ncentroids):
                start_ix = 0
                for ix, end_ix in enumerate(ndesc_by_img):
                    mask = I[start_ix:end_ix] == cluster
                    database[ix, cluster, :] = \
                        xtrain[start_ix:end_ix][mask, :].sum(axis=0)
                    start_ix = end_ix
            database = database.reshape(nimg, ncentroids * dim)
            database = sanitize(database)
            index.add(database)

            # reset
            ndesc_in_batch = 0
            xtrain, database, ndesc_by_img = [], [], []


    # last batch
    xtrain = sanitize(np.concatenate(xtrain))
    _, I = kmeans.index.search(xtrain, 1)
    _ = None
    I = I.flatten()

    nimg = len(ndesc_by_img)
    # vlad
    xtrain = xtrain - kmeans.centroids[I]
    database = np.empty((nimg, ncentroids, dim))
    for cluster in range(ncentroids):
        start_ix = 0
        for ix, end_ix in enumerate(ndesc_by_img):
            mask = I[start_ix:end_ix] == cluster
            database[ix, cluster, :] = \
                xtrain[start_ix:end_ix][mask, :].sum(axis=0)
            start_ix = end_ix
    database = database.reshape(nimg, ncentroids * dim)
    database = sanitize(database)
    index.add(database)

    # reset
    ndesc_in_batch = 0
    xtrain, database, ndesc_by_img = [], [], []


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
        
        # record the index of the 
        filenames_query.append(filename)
        filenames_query_ix.append(desc.shape[0])
        
        # print status
        total_descriptors += desc.shape[0]
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
    _, I = kmeans.index.search(xtest, 1)
    _ = None
    I = I.flatten()

    nimg = len(filenames_query)
    # vlad
    xtest = xtest - kmeans.centroids[I]
    query_database = np.empty((nimg, ncentroids, dim))
    for cluster in range(ncentroids):
        start_ix = 0
        for ix, end_ix in enumerate(filenames_query_ix):
            mask = I[start_ix:end_ix] == cluster
            query_database[ix, cluster, :] = \
                xtest[start_ix:end_ix][mask, :].sum(axis=0)
            start_ix = end_ix
    query_database = query_database.reshape(nimg, ncentroids * dim)
    query_database = sanitize(query_database)


    #########################
    #####    PREDICT    #####
    #########################

    print('______ Prediction ______ ', flush=True)

    print(index.ntotal)
    print(len(filenames_index))

    k = 100
    # Search on CPU
    D, I = index.search(query_database, k)

    # ## Search on GPU
    # res = faiss.StandardGpuResources()
    # gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    # D, I = gpu_index.search(query_database, k)
    # D = None

    # results = {}
    # for query_ix, query_result in enumerate(I):
    #     results[filenames_query[query_ix]] = ' '.join(
    #         [filenames_index[ix] for ix in query_result if ix != -1])

    results = {}
    for query_ix, query_result in enumerate(I):
        res = []
        for img_ix in query_result:
            if img_ix != -1:
                try:
                    filename = filenames_index[img_ix]
                    res.append(filename)
                except IndexError:
                    print('img ix', img_ix)
        try:
            query_filename = filenames_query[query_ix]
        except IndexError:
            print('query_ix', query_ix)
        results[query_filename] = ' '.join(res)


    submit = pd.read_csv(sample_submission)
    submit['images'] = submit['id'].apply(lambda ix: results.get(ix, ''))

    submit_filename = 'submit_{:%Y-%m-%d}_vlad_full_index.csv.gz'.format(
        datetime.now())
    submit.to_csv(submit_filename, index=False, compression='gzip')


if __name__ == '__main__':
    main()