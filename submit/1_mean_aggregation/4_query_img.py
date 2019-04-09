import sys
import glob
import time
import _pickle as pickle
from os.path import splitext, basename, join, isfile
from datetime import datetime

sys.path.append('/home/alexandrearaujo/library/faiss/')
import pandas as pd
import numpy as np
import faiss
from delf import feature_io

_STATUS_CHECK_ITERATIONS = 10000

kaggle_folder = '/home/alexandrearaujo/kaggle/landmark-retrieval-challenge'
storage_folder = '/media/hdd1/kaggle/landmark-retrieval-challenge/'

data_folder = join(kaggle_folder, 'data')
submit_folder = join(kaggle_folder, 'submit')
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


def load_mean_agg(path_list, dim):
    num_images = len(path_list)
    data = np.zeros((num_images, dim))
    filenames = []        
    start = time.clock()
    for i, path in enumerate(path_list):
        filename = splitext(basename(path))[0]
        if i % _STATUS_CHECK_ITERATIONS == 0 and i != 0:
            elapsed = (time.clock() - start)
            print('Processing image {} out of {}, last'
                    ' {} images took {:.5f} seconds'.format(i, num_images,
                          _STATUS_CHECK_ITERATIONS, elapsed), flush=True)
            start = time.clock()
        try:
            loc, _, desc, _, _ = feature_io.ReadFromFile(path)
        except Exception as error:
            print('problem with file {}. error : {}'.format(
                            filename, error), flush=True)
        # fill the database
        data[i, :] = np.mean(desc, axis=0)
        filenames.append(filename)
    return sanitize(data), np.array(filenames)

def main():

    dim = 40 # dim of descriptors

    index_files = glob.glob(join(data_folder, 'feature_index_256x256', '*'))
    query_files = glob.glob(join(data_folder, 'feature_test_256x256', '*'))

    xdatabase, filenames_index = load_mean_agg(index_files, dim)
    write_memmap_file(join(data_folder, 'xdatabase_mean_agg.bin'), xdatabase)
    pickle_dump(filenames_index, join(data_folder, 'filename_index.bin'))

    xquery, filenames_query = load_mean_agg(query_files, dim)
    write_memmap_file(join(data_folder, 'xquery_mean_agg.bin'), xquery)
    pickle_dump(filenames_query, join(data_folder, 'filename_query.bin'))


    # xdatabase = read_memmap_file(join(data_folder, 'xdatabase_mean_agg.bin')).reshape(-1, 40)
    # xquery = read_memmap_file(join(data_folder, 'xquery_mean_agg.bin')).reshape(-1, 40)

    # filenames_index = pickle_load(join(data_folder, 'filename_index.bin'))
    # filenames_query = pickle_load(join(data_folder, 'filename_query.bin'))


    # faiss
    nlist = 100
    m = 8
    quantizer = faiss.IndexFlatL2(dim)  # this remains the same
    # 8 specifies that each sub-vector is encoded as 8 bits
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8) 

    index.train(xdatabase)
    index.add(xdatabase)

    k = 100
    D, I = index.search(xquery, k)

    results = {}
    for query_ix, query_result in enumerate(I):
        l = [filenames_index[ix] for ix in query_result if ix != -1]
        results[filenames_query[query_ix]] = ' '.join(l)

    submit = pd.read_csv(sample_submission)
    submit['images'] = submit['id'].apply(lambda ix: results.get(ix, ''))

    submit_filename = 'submit_{:%Y-%m-%d}_mean_agg.csv.gz'.format(datetime.now())
    path = join(submit_folder, submit_filename)
    submit.to_csv(path, index=False, compression='gzip')


if __name__ == '__main__':
    main()