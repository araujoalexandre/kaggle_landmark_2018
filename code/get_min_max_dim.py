
import sys
import glob
from os.path import splitext, basename

sys.path.append('/home/alexandrearaujo/library/faiss/')
import faiss
from delf import feature_io
import numpy as np

query_files = glob.iglob('/media/hdd1/kaggle/landmark-retrieval-challenge/feature_test_256x256/*')
index_files = glob.iglob('/media/hdd1/kaggle/landmark-retrieval-challenge/feature_index_256x256/*')

results = {}
min_dim = np.iinfo(np.int32).max
max_dim = np.iinfo(np.int32).min
for i, path_index in enumerate(index_files):
    if i % 100000 == 0: print(i)
    filename = splitext(basename(path_index))[0]
    try:
        index_loc, _, index_desc, _, _ = feature_io.ReadFromFile(path_index)
    except Exception as error:
        print('problem with file {}. error : {}'.format(filename, error))
    min_dim = np.min([index_desc.shape[0], min_dim])
    max_dim = np.max([index_desc.shape[0], max_dim])
print('index', min_dim, max_dim)

min_dim = np.iinfo(np.int32).max
max_dim = np.iinfo(np.int32).min
for i, path_index in enumerate(query_files):
    if i % 100000 == 0: print(i)
    filename = splitext(basename(path_index))[0]
    try:
        index_loc, _, index_desc, _, _ = feature_io.ReadFromFile(path_index)
    except Exception as error:
        print('problem with file {}. error : {}'.format(filename, error))
    min_dim = np.min([index_desc.shape[0], min_dim])
    max_dim = np.max([index_desc.shape[0], max_dim])
print('test', min_dim, max_dim)


# index 1 425
# test 1 386
