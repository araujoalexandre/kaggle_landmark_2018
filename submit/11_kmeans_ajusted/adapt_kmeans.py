  

import warnings
warnings.filterwarnings("ignore")

import sys
import glob
import time
import gc
import copy
import argparse
import pickle
import tarfile
from collections import defaultdict
from multiprocessing import Queue, JoinableQueue
from threading import Thread
from os.path import splitext, basename, join, isfile, dirname, realpath, exists
from datetime import datetime

import numpy as np

# custom class 
from kmeans import Kmeans


kaggle_folder = '/home/alexandrearaujo/kaggle/landmark-retrieval-challenge'
valid_folder = join(kaggle_folder, 'validation')
storage_folder = '/media/hdd1/kaggle/landmark-retrieval-challenge/'


def pickle_load(path):
  """function to load pickle object"""
  with open(path, 'rb') as f:
    return pickle.load(f, encoding='latin1')

def pickle_dump(file, path, force=False):

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

  """dump a file without deleting an existing one"""
  if force:
    _pickle_dump(file, path)
  elif not force:
    new_path = _get_new_name(path)
    _pickle_dump(file, new_path)



queue = Queue(1000000)

class LoadThread(Thread):
    
  def __init__(self, train_iterator, _STATUS_CHECK_ITERATIONS):
    super(LoadThread, self).__init__()
    self.train_iterator = train_iterator
    self._STATUS_CHECK_ITERATIONS = _STATUS_CHECK_ITERATIONS

  def run(self):
    global queue
    start = datetime.now()
    for iteration, filename, desc in self.train_iterator:
      queue.put((iteration, filename, desc))
      if iteration % self._STATUS_CHECK_ITERATIONS == 0 and iteration != 0:
        elapsed = (datetime.now() - start).total_seconds()
        print('iteration {}, load last batch took {:.2f}, queue size {}'.format(
          iteration, elapsed, queue.qsize()))
        start = datetime.now()
    queue.put((None, None, None))
    print('load thread done', flush=True)
    return

class ProcessThread(Thread):
    
  def __init__(self, dim, ncentroids, kmeans, nimg_train, 
                _STATUS_CHECK_ITERATIONS):
    super(ProcessThread, self).__init__()
    self.dim = dim
    self.ncentroids = ncentroids
    self.kmeans = kmeans
    self.nimg_train = nimg_train
    self._STATUS_CHECK_ITERATIONS = _STATUS_CHECK_ITERATIONS

    self.kmeans_centroid_adapt = np.zeros((self.ncentroids, self.dim))
    self.count_desc = defaultdict(int)

  def _sanitize(self, data):
    """ convert array to a c-contiguous float array """
    return np.ascontiguousarray(data.astype('float32'))
  
  def run(self):
    global queue
    time.sleep(5)
    self.start = datetime.now()
    total_descriptors = 0
    while True:

      data, n_desc = [], []
      for x in range(2000):
        iteration, filename, desc = queue.get()
        if iteration is None: 
          break
        # count the number of descriptors processed and print status
        total_descriptors += desc.shape[0]
        # print status
        if iteration % self._STATUS_CHECK_ITERATIONS == 0 and iteration != 0:
          elapsed = (datetime.now() - self.start).total_seconds()
          print('Loading image {} out of {}, last {} images took {:.5f}'
              ' seconds, total number of descriptors {}'.format(
                iteration, self.nimg_train, self._STATUS_CHECK_ITERATIONS, 
                elapsed, total_descriptors), flush=True)
          self.start = datetime.now()
        data.append(desc)
      
      if iteration is None: 
        break
      
      # process data
      data = self._sanitize(np.concatenate(data))
      predicted = self.kmeans.assign(data)[1].flatten()
      for cluster in range(self.ncentroids):
        mask = predicted == cluster
        self.kmeans_centroid_adapt[cluster, :] = data[mask, :].sum(axis=0)
        self.count_desc[cluster] += np.sum(mask)

    # finalize
    for cluster in range(self.ncentroids):
      self.kmeans_centroid_adapt[cluster] = \
        self.kmeans_centroid_adapt[cluster] / self.count_desc[cluster]
    print('process thread done', flush=True)
    return 

  def join(self):
    Thread.join(self)
    return self.kmeans_centroid_adapt



class AdaptKmeans:

  def __init__(self, debug=False, gpu=False):

    self.cur_dir = dirname(realpath(__file__))
    self.gpu = gpu

    self.train_files = glob.glob(join(storage_folder, 
      'feature_index_rescale_delf_pkl', '*'))

    self.dim = 40 # dim of descriptors

    self.train_filenames = []

    if debug:
      self._init_debug_mode()
    else:
      self._init()

  def _init(self):
    self._STATUS_CHECK_ITERATIONS = 10000
    self._BATCH_TRAIN = 40000000

    self.nimg_train = len(self.train_files)

    # params
    self.ncentroids = 128
    niter = 50
    verbose = True
    max_points_per_centroid = 10000000

    # init kmeans
    self.kmeans = Kmeans(self.dim, self.ncentroids, 
            niter=niter, verbose=verbose, 
            max_points_per_centroid=max_points_per_centroid, 
            gpu=self.gpu)
    # check if kmeans centroid already trained
    kmeans_checkpoint = join(self.cur_dir, 'kmeans.dump')
    self.kmeans.load(kmeans_checkpoint)

  def _init_debug_mode(self):
    self.train_files = self.train_files[:30000]
    self.nimg_train = len(self.train_files)
    self._BATCH_TRAIN = 1000000
    
    self._STATUS_CHECK_ITERATIONS = 1000
    
    # params
    self.ncentroids = 128
    niter = 50
    verbose = True
    
    # init kmeans
    self.kmeans = Kmeans(self.dim, self.ncentroids, 
            niter=niter, verbose=verbose, 
            max_points_per_centroid=None, 
            gpu=self.gpu)
    # check if kmeans centroid already trained
    kmeans_checkpoint = join(self.cur_dir, 'kmeans.dump')
    self.kmeans.load(kmeans_checkpoint)

  def _images_iterator(self, images_path):
    for iteration, path in enumerate(images_path):
      filename = splitext(basename(path))[0]
      loc, desc = pickle_load(path)
      if not all([isinstance(filename, str), isinstance(desc, np.ndarray)]):
        print('file {} is corrupted'.format(filename))
        filename = ''
        desc = np.zeros((1, 128))
      yield iteration, filename, desc

  def _print_status(self, iteration, num_images, total_descriptors):
    # print status
    if iteration % self._STATUS_CHECK_ITERATIONS == 0 and iteration != 0:
      elapsed = (datetime.now() - self.start).total_seconds()
      print('Loading image {} out of {}, last {} images took {:.5f}'
          ' seconds, total number of descriptors {}'.format(
            iteration, num_images, self._STATUS_CHECK_ITERATIONS, 
            elapsed, total_descriptors), flush=True)
      self.start = datetime.now()

  def compute_adapt_kmeans(self):

    print('reaload every thing to adjust kmeans', flush=True)
    # I/O workload
    loader = LoadThread(self._images_iterator(self.train_files), 
              self._STATUS_CHECK_ITERATIONS)
    
    # heavy workload
    # the workload is not that heavy, the LoadThread will be way behind
    process = ProcessThread(self.dim, self.ncentroids, self.kmeans, 
                  self.nimg_train, self._STATUS_CHECK_ITERATIONS)

    # start work
    loader.start()
    process.start()
    # we get the index back 
    self.kmeans_centroid_adapt = process.join()

    print('saving kmeans_adjusted', flush=True)
    pickle_dump(self.kmeans_centroid_adapt, 'kmeans_adjusted.dump')



def main():
  
  parser = argparse.ArgumentParser(description="AdaptKmeans")
  parser.add_argument("--debug", type=bool, default=False)
  parser.add_argument("--gpu",  type=bool, default=False)
  args = parser.parse_args()

  kaggle = AdaptKmeans(debug=args.debug, gpu=args.gpu)
  kaggle.compute_adapt_kmeans()
  print('done', flush=True)

if __name__ == '__main__':
  main()
