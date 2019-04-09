
import warnings
warnings.filterwarnings("ignore")

import sys
import glob
import time
import gc
import copy
import argparse
import _pickle as pickle
from multiprocessing import Pool
from collections import defaultdict, Counter
from os.path import splitext, basename, join, isfile, dirname, realpath, exists
from datetime import datetime

sys.path.append('/home/alexandrearaujo/libraries/faiss/')
import pandas as pd
import numpy as np
import faiss
from delf import feature_io


kaggle_folder = '/home/alexandrearaujo/kaggle/landmark-retrieval-challenge'
storage_folder = '/media/hdd1/kaggle/landmark-retrieval-challenge/'

# def pickle_load(path):
#   """function to load pickle object"""
#   with open(path, 'rb') as f:
#     return pickle.load(f, encoding='latin1')

def pickle_load(path):
  """function to load pickle object"""
  with open(path, 'rb') as f:
    return pickle.load(f)

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


class Test:

  def __init__(self, debug=False, gpu=False):

    self.cur_dir = dirname(realpath(__file__))

    self.gpu = gpu

    self._STATUS_CHECK_ITERATIONS = 10000
    self._BATCH_TRAIN = 30000000

    self.images_folder = join(kaggle_folder, 'img')
    self.sample_submission = join(kaggle_folder, 'data', 
      'sample_submission.csv')

    self.train_files = glob.glob(join(storage_folder, 
      'feature_index_rescale_delf_pkl', '*'))
    self.test_files = glob.glob(join(storage_folder, 
      'feature_test_rescale_delf_pkl', '*'))

    np.random.shuffle(self.train_files)

    self._images_iterator = self._images_iterator_pickle

    self.nimg_train = len(self.train_files)
    self.nimg_test = len(self.test_files)

    self.train_filenames = []
    self.test_filenames = []
    self.test_ndesc_by_img = []

    # params
    self.dim = 40 # dim of descriptors
    self.ncentroids = 128
    niter = 100
    verbose = True
    max_points_per_centroid = 10000000

    if debug:
      self.train_files = self.train_files[:5000]
      self.test_files = self.test_files[:1000]
      self.nimg_train = len(self.train_files)
      self.nimg_test = len(self.test_files)
      self._BATCH_TRAIN = 2000000
      self._STATUS_CHECK_ITERATIONS = 1000
      # self._make_results = self._make_results_debug

  def _sanitize(self, data):
    """ convert array to a c-contiguous float array """
    return np.ascontiguousarray(data.astype('float32'))

  def _images_iterator_pickle(self, images_path):
    for iteration, path in enumerate(images_path):
      # filename = splitext(basename(path))[0]
      filename = 0
      loc, desc = pickle_load(path)
      yield iteration, filename, loc, desc

  def _print_status(self, iteration, num_images, total_descriptors):
    # print status
    if iteration % self._STATUS_CHECK_ITERATIONS == 0 and iteration != 0:
      elapsed = (datetime.now() - self.start).total_seconds()
      print('Loading image {} out of {}, last {} images took {:.5f}'
          ' seconds, total number of descriptors {}'.format(
            iteration, num_images, self._STATUS_CHECK_ITERATIONS, 
            elapsed, total_descriptors), flush=True)
      self.start = datetime.now()

  def load_train_by_batch(self):

    print('loading database to compute index', flush=True)
    xtrain = []
    ndesc_by_img = []
    self.start = datetime.now()
    total_descriptors = 0
    ndesc_in_batch = 0
    
    # loop over all images
    for iteration, filename, loc, desc in \
        self._images_iterator(self.train_files):

      # count the number of descriptors processed and print status
      total_descriptors += desc.shape[0]
      self._print_status(iteration, self.nimg_train, total_descriptors)

      # accumulate
      if ndesc_in_batch < self._BATCH_TRAIN:
        # fill the database
        xtrain.append(desc)
        # batch variables 
        ndesc_by_img.append(desc.shape[0])
        # ndesc_in_batch the number of descriptors 
        ndesc_in_batch += desc.shape[0]
        # record filenames of images
        self.train_filenames.append(filename)
        continue


      # add index
      if ndesc_in_batch >= self._BATCH_TRAIN:
        print('total number of descriptors reached, '
            'adding descriptors to index', flush=True)
        
        xtrain = self._sanitize(np.concatenate(xtrain))
        print('kmeans assignement on xtrain', flush=True)

        # reset
        ndesc_in_batch = 0
        xtrain, database, ndesc_by_img = [], [], []

if __name__ == '__main__':
    t = Test(debug=True)
    t.load_train_by_batch()