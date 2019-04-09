
import warnings
warnings.filterwarnings("ignore")

import sys
import glob
import time
import gc
import _pickle as pickle
from collections import defaultdict, Counter
from os.path import splitext, basename, join, isfile
from datetime import datetime

import numpy as np
from delf import feature_io

kaggle_folder = '/home/alexandrearaujo/kaggle/landmark-retrieval-challenge'
storage_folder = '/media/hdd1/kaggle/landmark-retrieval-challenge/'

data_folder = join(kaggle_folder, 'data')
img_folder = join(kaggle_folder, 'img')
sample_submission = join(kaggle_folder, 'data', 'sample_submission.csv')

_STATUS_CHECK_ITERATIONS = 10000

def run():

    index_files = glob.glob(join(img_folder, 'feature_index_256x256', '*'))
    query_files = glob.glob(join(img_folder, 'feature_test_256x256', '*'))

    num_images = len(index_files)
    start = time.clock()
    total_descriptors = 0
    print('______ Loading Train ______ ')
    for i, path in enumerate(index_files):
        filename = splitext(basename(path))[0]
        try:
            loc, _, desc, _, _ = feature_io.ReadFromFile(path)
        except Exception as error:
            print('problem with file {}. error : {}'.format(
                            filename, error), flush=True)
        # count the number of descriptors 
        total_descriptors += desc.shape[0]
        # print status
        if i % _STATUS_CHECK_ITERATIONS == 0 and i != 0:
            elapsed = (time.clock() - start)
            print('Loading image {} out of {}, last {} images took {:.5f}'
                    ' seconds, total number of descriptors {}'.format(
                        i, num_images, _STATUS_CHECK_ITERATIONS, 
                        elapsed, total_descriptors), flush=True)
            start = time.clock()

    num_images = len(index_files)
    start = time.clock()
    total_descriptors = 0
    print('______ Loading Test ______ ')
    for i, path in enumerate(query_files):
        filename = splitext(basename(path))[0]
        try:
            loc, _, desc, _, _ = feature_io.ReadFromFile(path)
        except Exception as error:
            print('problem with file {}. error : {}'.format(
                            filename, error), flush=True)
        # count the number of descriptors 
        total_descriptors += desc.shape[0]
        # print status
        if i % _STATUS_CHECK_ITERATIONS == 0 and i != 0:
            elapsed = (time.clock() - start)
            print('Loading image {} out of {}, last {} images took {:.5f}'
                    ' seconds, total number of descriptors {}'.format(
                        i, num_images, _STATUS_CHECK_ITERATIONS, 
                        elapsed, total_descriptors), flush=True)
            start = time.clock()


if __name__ == '__main__':
    run()