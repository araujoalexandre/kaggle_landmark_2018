"""
Image retrieval system 

Extraction of sift local descriptors from images with opencv  
local descriptors are aggreagated with VLAD [2]
VLAD Descriptors are normalize with SSR [3]
Retrieval system made with Faiss [4]

[2] Aggregating local descriptors into a compact image representation.
H. Jégou, M. Douze, C. Schmid, and P. Pérez. 

[3] Aggregating local image descriptors into compact codes
Hervé Jégou, Florent Perronnin, Matthijs Douze, Jorge Sánchez, Patrick
Pérez, Cordelia Schmid

[4] Billion-scale similarity search with GPUs
Jeff Johnson, Matthijs Douze, Hervé Jégou
https://github.com/facebookresearch/faiss

"""

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

sys.path.append('/home/alexandrearaujo/libraries/faiss/')
import pandas as pd
import numpy as np
import faiss
from delf import feature_io
from scipy.cluster.vq import whiten
import cv2
from scipy.stats import multivariate_normal

# custom class 
from kmeans import Kmeans


kaggle_folder = '/home/alexandrearaujo/kaggle/landmark-retrieval-challenge'
valid_folder = join(kaggle_folder, 'validation')
storage_folder = '/media/hdd1/kaggle/landmark-retrieval-challenge/'

class Timer:
    
    def __init__(self, msg):
        self.msg = msg
    
    def __enter__(self):
        self.start = datetime.now()
        
    def __exit__(self, *args):
        elapsed = (datetime.now() - self.start).total_seconds()
        print('{} on last batch took {:.3f}'.format(self.msg, elapsed),
              flush=True)


def generate_gmm(descriptors, k):
  # Throw away gaussians with weights that are too small:
  em = cv2.ml.EM_create()
  em.setClustersNumber(k)
  em.trainEM(descriptors)

  means = np.float32(em.getMeans())
  covs = np.float32(em.getCovs())
  weights = np.float32(em.getWeights())[0]

  th = 1.0 / k
  means   = np.array([m for w, m in zip(weights, means) if w > th])
  covs    = np.array([c for w, c in zip(weights, covs) if w > th])
  weights = np.array([w for w in weights if w > th])
  return means, covs, weights

def likelihood_moment(x, ytk, moment):  
  x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
  return x_moment * ytk

def likelihood_statistics(samples, means, covs, weights):
  gaussians = {}
  s0, s1, s2 = defaultdict(float), defaultdict(float), defaultdict(float)
  samples = list(zip(range(0, len(samples)), samples))
  
  g = [multivariate_normal(mean=means[k], cov=covs[k]) for k in range(0, len(weights)) ]
  for index, x in samples:
    gaussians[index] = np.array([g_k.pdf(x) for g_k in g])

  for k in range(0, len(weights)):
    s0[k], s1[k], s2[k] = 0, 0, 0
    for index, x in samples:
      probabilities = np.multiply(gaussians[index], weights)
      probabilities = probabilities / np.sum(probabilities)
      s0[k] += likelihood_moment(x, probabilities[k], 0)
      s1[k] += likelihood_moment(x, probabilities[k], 1)
      s2[k] += likelihood_moment(x, probabilities[k], 2)

  return s0, s1, s2

def fisher_vector_weights(s0, s1, s2, means, covs, w, T):
  return np.array([((s0[k] - T * w[k]) / np.sqrt(w[k]) ) for k in range(0, len(w))])

def fisher_vector_means(s0, s1, s2, means, sigma, w, T):
  return np.array([(s1[k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])

def fisher_vector_sigma(s0, s1, s2, means, sigma, w, T):
  return np.array([(s2[k] - 2 * means[k]*s1[k]  + (means[k]*means[k] - sigma[k]) * s0[k]) / (np.sqrt(2*w[k])*sigma[k])  for k in range(0, len(w))])

def normalize(fisher_vector):
  v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
  return v / np.sqrt(np.dot(v, v))

def fisher_vector(samples, means, covs, weights):
  s0, s1, s2 =  likelihood_statistics(samples, means, covs, weights)
  covs = np.float32([np.diagonal(covs[k]) for k in range(0, covs.shape[0])])
  a = fisher_vector_weights(s0, s1, s2, means, covs, weights, samples.shape[0])
  b = fisher_vector_means(s0, s1, s2, means, covs, weights, samples.shape[0])
  c = fisher_vector_sigma(s0, s1, s2, means, covs, weights, samples.shape[0])
  fv = np.concatenate([a.flatten(), b.flatten(), c.flatten()])
  fv = normalize(fv)
  return fv[np.newaxis]







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


class KaggleRetrievalTools: 

  def _print_status(self, iteration, num_images, total_descriptors):
    # print status
    if iteration % self._STATUS_CHECK_ITERATIONS == 0 and iteration != 0:
      elapsed = (datetime.now() - self.start).total_seconds()
      print('Loading image {} out of {}, last {} images took {:.5f}'
          ' seconds, total number of descriptors {}'.format(
            iteration, num_images, self._STATUS_CHECK_ITERATIONS, 
            elapsed, total_descriptors), flush=True)
      self.start = datetime.now()

  def _sanitize(self, data):
    """ convert array to a c-contiguous float array """
    return np.ascontiguousarray(data.astype('float32'))

  def _make_vlad_by_batch(self, x, n_desc, predicted):
    desc_cumsum = np.cumsum(n_desc)
    img_indexes = list(zip([0] + list(desc_cumsum[:-1]), desc_cumsum))
    database = np.zeros((len(n_desc), self.ncentroids * self.dim))
    for ix, (start_ix, end_ix) in enumerate(img_indexes):
      x_subset = x[start_ix:end_ix]
      predicted_susbset = predicted[start_ix:end_ix]
      database[ix, :] = self._make_vlad(x_subset, predicted_susbset)
    database = self._sanitize(database)
    return database

  def _make_vlad(self, x, predicted):
    predicted = predicted.flatten()
    x = x - self.kmeans.centroids[predicted]
    x /= np.linalg.norm(x, ord=1, axis=1)[..., np.newaxis]
    vlad = np.zeros((self.ncentroids, self.dim))
    for cluster in range(self.ncentroids):
      mask = predicted == cluster
      if np.sum(mask):
        vlad[cluster, :] = x[mask, :].sum(axis=0)
    # reshape
    vlad = vlad.reshape(self.ncentroids * self.dim)
    # power normalization, also called square-rooting normalization
    vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
    # L2 normalization
    vlad = vlad / np.linalg.norm(vlad)
    return vlad[np.newaxis]

  
  def _make_fisher_by_batch(self, x, n_desc, gmm):
    desc_cumsum = np.cumsum(n_desc)
    img_indexes = list(zip([0] + list(desc_cumsum[:-1]), desc_cumsum))
    database = []
    for ix, (start_ix, end_ix) in enumerate(img_indexes):
      x_subset = x[start_ix:end_ix]
      database.append(self._make_fisher(x_subset, gmm))
    database = self._sanitize(np.concatenate(database))
    return database

  def _make_fisher(self, x, gmm):
    return fisher_vector(x, *gmm)





class KaggleRetrievalDebug:

  def _init_debug_mode(self):

    self.train_files = self.train_files[:80]
    self.test_files = self.test_files[:10]
    self.nimg_train = len(self.train_files)
    self.nimg_test = len(self.test_files)
    self._BATCH_TRAIN = 30000
    
    self._STATUS_CHECK_ITERATIONS = 1000
    self._make_results = self._make_results_debug
    
    # params
    self.ncentroids = 8
    niter = 50
    verbose = True
    
    # init kmeans
    self.kmeans = Kmeans(self.dim, self.ncentroids, 
            niter=niter, verbose=verbose, 
            max_points_per_centroid=None, 
            gpu=self.gpu)
    self._try_load_kmeans()
    
    # init index
    self._try_load_index()
    
    # init train generator
    self.train_iterator = self._images_iterator(self.train_files)

  def _make_results_debug(self, predicted):
    results = {}
    for query_ix, query_result in enumerate(predicted):
      res = []
      for img_ix in query_result:
        if img_ix != -1:
          try:
            filename = self.train_filenames[img_ix]
            res.append(filename)
          except IndexError:
            print('img ix', img_ix)
      try:
        query_filename = self.test_filenames[query_ix]
      except IndexError:
        print('query_ix', query_ix)
      results[query_filename] = ' '.join(res)
    return results


class KaggleRetrievalValidation:

  def _init_validation(self):
    self.target_dict = pickle_load(join(valid_folder, 'target.pkl'))
    self.target_train = pickle_load(join(valid_folder, 'target_train.pkl'))

    self.train_files = glob.glob(join(valid_folder, 'feature_train_delf', '*'))
    self.test_files = glob.glob(join(valid_folder, 'feature_test_delf', '*'))

    self._STATUS_CHECK_ITERATIONS = 1000
    self._BATCH_TRAIN = 10000

    self.nimg_train = len(glob.glob(join(valid_folder, 'train', '*')))
    self.nimg_test = len(glob.glob(join(valid_folder, 'test', '*')))

    # params
    self.ncentroids = 128
    niter = 30
    verbose = True
    max_points_per_centroid = 1000

    # init kmeans
    self.kmeans = Kmeans(self.dim, self.ncentroids, 
            niter=niter, verbose=verbose, 
            max_points_per_centroid=max_points_per_centroid, 
            gpu=self.gpu)
    self._try_load_kmeans()

    # init index
    self._try_load_index()

    # init train generator
    self.train_iterator = self._images_iterator(self.train_files)

  def map100metric(self, predicted, actual):

    actual = actual.reshape(-1, 1)
    metric = 0.
    for i in range(100):
        metric += np.sum(actual == predicted[:, i]) / (i + 1)
    metric /= actual.shape[0]
    return metric

  def predict_validation(self, query_database):
    print('compute score', flush=True)
    k = 100
    if self.gpu:
      ## Search on GPU
      res = faiss.StandardGpuResources()
      gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)
      predicted = gpu_index.search(query_database, k)[1]
    else:
      # Search on CPU
      predicted = self.index.search(query_database, k)[1]
    
    target = []
    for i, query in enumerate(query_database):
      target.append(self.target_dict[self.test_filenames[i]])
    target = np.array(target)

    predicted_map = np.zeros_like(predicted)
    for i, j in np.ndindex(predicted.shape):
      if predicted[i, j] == -1:
        predicted_map[i, j] = -1
        continue
      name = self.train_filenames[predicted[i, j]]
      predicted_map[i, j] = self.target_train[name]

    score = self.map100metric(predicted_map, target)
    print('map100 = {}'.format(score))




queue = Queue(1000000)

class LoadThread(Thread, KaggleRetrievalTools):
    
  def __init__(self, train_iterator):
    super(LoadThread, self).__init__()
    self.train_iterator = train_iterator

  def run(self):
    global queue
    start = datetime.now()
    for iteration, filename, desc in self.train_iterator:
      queue.put((iteration, filename, desc))
      if iteration % 10000 == 0:
        elapsed = (datetime.now() - start).total_seconds()
        print('load last batch took {:.2f}, queue size {}'.format(
          elapsed, queue.qsize()))
        start = datetime.now()
    queue.put((None, None, None))
    return

class ProcessThread(Thread, KaggleRetrievalTools):
  
  def __init__(self, dim, ncentroids, train_filenames, kmeans, index, 
                total_descriptors, nimg_train, _STATUS_CHECK_ITERATIONS, gmm):
    super(ProcessThread, self).__init__()
    self.dim = dim
    self.ncentroids = ncentroids
    self.train_filenames = train_filenames
    self.kmeans = kmeans
    self.index = index
    self.total_descriptors = total_descriptors
    self.nimg_train = nimg_train
    self._STATUS_CHECK_ITERATIONS = _STATUS_CHECK_ITERATIONS
    self.gmm = gmm

  # def _process_data(self, iteration, n_desc, data):
  #     predicted = self.kmeans.assign(data, nclusters=1)[1]
  #     database = self._make_vlad_by_batch(data, n_desc, predicted)
  #     self.index.add(self._sanitize(database))

  def _process_data(self, iteration, n_desc, data,):
      database = self._make_fisher_by_batch(data, n_desc, self.gmm)
      self.index.add(self._sanitize(database))
  
  def run(self):
    global queue
    # time.sleep(300)
    self.start = datetime.now()
    while True:

      data, n_desc = [], []
      for x in range(2000):
        iteration, filename, desc = queue.get()
        if iteration is None: 
          break
        self.train_filenames.append(filename)
        n_desc.append(desc.shape[0])
        # count the number of descriptors processed and print status
        self.total_descriptors += desc.shape[0]
        self._print_status(iteration, self.nimg_train, self.total_descriptors)
        # aggregate descriptors
        data.append(desc)
      
      if iteration is None: 
        break
      
      # process data
      data = self._sanitize(np.concatenate(data))      
      self._process_data(iteration, n_desc, data)
      
  def join(self):
    Thread.join(self)
    return self.index, self.train_filenames



class KaggleRetrieval(KaggleRetrievalTools, KaggleRetrievalDebug, 
                        KaggleRetrievalValidation):

  def __init__(self, debug=False, gpu=False, valid=True):

    self.cur_dir = dirname(realpath(__file__))
    self.gpu = gpu

    self.images_folder = join(kaggle_folder, 'img')
    self.sample_submission = join(kaggle_folder, 'data', 
      'sample_submission.csv')
    
    self.train_files = glob.glob(join(storage_folder, 
      'feature_index_rescale_delf_pkl', '*'))
    self.test_files = glob.glob(join(storage_folder, 
      'feature_test_rescale_delf_pkl', '*'))

    self.gaussians_mixture = False

    self.dim = 40 # dim of descriptors

    self.train_filenames = []
    self.test_filenames = []

    if debug:
      self._init_debug_mode()
    elif valid:
      self._init_validation()
      self.predict = self.predict_validation
    else:
      self._init()

  def _init(self):

    self._STATUS_CHECK_ITERATIONS = 10000
    self._BATCH_TRAIN = 10000 # numbers of images

    self.nimg_train = len(self.train_files)
    self.nimg_test = len(self.test_files)

    # params
    self.ncentroids = 128
    niter = 50
    verbose = True
    max_points_per_centroid = 100000

    # init kmeans
    self.kmeans = Kmeans(self.dim, self.ncentroids, 
            niter=niter, verbose=verbose, 
            max_points_per_centroid=max_points_per_centroid, 
            gpu=self.gpu)
    self._try_load_kmeans()

    # init index
    self._try_load_index()

    # init train generator
    self.train_iterator = self._images_iterator(self.train_files)

  def _try_load_kmeans(self):
    # check if kmeans centroid already trained
    kmeans_checkpoint = join(self.cur_dir, 'kmeans.dump')
    if exists(kmeans_checkpoint):
      print('kmeans centroids found, kmeans training not necessary')
      self.kmeans.load(kmeans_checkpoint)

  def _try_load_index(self):
    # check if index.dump exists
    index_checkpoint_full = join(self.cur_dir, 'index_full.dump')
    index_checkpoint = join(self.cur_dir, 'index_trained.dump')
    
    # try to load full index first
    if exists(index_checkpoint_full):
      print('full index checkpoint found, loading database not necessary')
      self.index = faiss.read_index(index_checkpoint_full)
      self.index_full_trained = True
      self.index_trained = True
      # if index_full exists filename should also exists
      self.train_filenames = pickle_load(join(self.cur_dir, 
                  'filenames_index.dump'))
    
    # fall back to trained index if it exists
    elif exists(index_checkpoint):
      print('trained index checkpoint found')
      self.index = faiss.read_index(index_checkpoint)
      self.index_trained = True
      self.index_full_trained = False

    # init new index
    else:
      self.index_trained = False
      self.index_full_trained = False
      # self.index = faiss.IndexFlatL2(self.ncentroids * self.dim)
      self.index = faiss.IndexFlatL2(81)

  # def _images_iterator(self, images_path):
  #   for iteration, path in enumerate(images_path):
  #     filename = splitext(basename(path))[0]
  #     loc, desc = pickle_load(path)
  #     if not all([isinstance(filename, str), isinstance(desc, np.ndarray)]):
  #       print('file {} is corrupted'.format(filename))
  #       filename = ''
  #       desc = np.zeros((1, 128))
  #     yield iteration, filename, desc

  def _images_iterator(self, images_path):
    for iteration, path in enumerate(images_path):
      filename = splitext(basename(path))[0]
      try:
        loc, _, desc, _, _ = feature_io.ReadFromFile(path)
      except Exception as error:
        print('problem with file {}. error : {}'.format(
                filename, error), flush=True)
        desc = np.zeros((1, 40))
      yield iteration, filename, desc

  def compute_kmeans_and_index(self):

    self.total_descriptors = 0

    if self.index_trained and self.kmeans.is_trained:
      print('kmeans and index already trained')
      if self.index_full_trained:
        return 
      # loop to advance the iterator until _BATCH_TRAIN reached
      for iteration, filename, desc in self.train_iterator:
        self.train_filenames.append(filename)
        # training
        if iteration == self._BATCH_TRAIN:
          return

    if not self.index_trained and not self.kmeans.is_trained:
      print('compute kmeans and index', flush=True)
    elif not self.index_trained and self.kmeans.is_trained:
      print('compute index', flush=True)

    xtrain = []
    ndesc_by_img = []
    self.start = datetime.now()
    ndesc_in_batch = 0

    # loop over images until _BATCH_TRAIN reached
    for iteration, filename, desc in self.train_iterator:

      # count the number of descriptors processed and print status
      self.total_descriptors += desc.shape[0]
      self._print_status(iteration, self.nimg_train, self.total_descriptors)

      # accumulate
      if iteration == self._BATCH_TRAIN:
        break

      # fill the database
      xtrain.append(desc)
      ndesc_by_img.append(desc.shape[0])
      # record filenames of images
      self.train_filenames.append(filename)


    print('total number of descriptors reached', flush=True)
    xtrain = self._sanitize(np.concatenate(xtrain))
    # pickle_dump(xtrain, 'xtrain.pkl')
    # pickle_dump(ndesc_by_img, 'ndesc_by_img.pkl')

    # xtrain = pickle_load('xtrain.pkl')
    # ndesc_by_img = pickle_load('ndesc_by_img.pkl')

    # # train kmeans if is not trained 
    # if not self.kmeans.is_trained:
    #   print('training kmeans and saving centroids', flush=True)
    #   self.kmeans.train(xtrain)
    #   self.kmeans.save(self.cur_dir)

    if not self.gaussians_mixture:
      # self.gmm = generate_gmm(xtrain, 5)
      self.gmm = pickle_load('gmm.pkl')
      database = self._make_fisher_by_batch(xtrain, ndesc_by_img, self.gmm)
      self.gaussians_mixture = True
      # pickle_dump(self.gmm, 'gmm.pkl')


    # train index if is not trained
    if not self.index_trained:
      # predicted = self.kmeans.assign(xtrain, nclusters=1)[1]
      # database = self._make_vlad_by_batch(xtrain, ndesc_by_img, predicted)
      print('training index', flush=True)
      self.index.train(database)
      self.index_trained = True
      self.index.add(database)
      print('training index done, saving index', flush=True)
      faiss.write_index(self.index, join(self.cur_dir, 'index_trained.dump'))

  def load_train(self):

    if self.index_full_trained and self.kmeans.is_trained:
      return 

    print('loading database to compute index', flush=True)
    xtrain = []
    ndesc_by_img = []
    self.start = datetime.now()
    ndesc_in_batch = 0
    
    # I/O workload
    loader = LoadThread(self.train_iterator)
    
    # heavy workload
    # the workload is not that heavy, the LoadThread will be way behind
    process = ProcessThread(self.dim, self.ncentroids, self.train_filenames, 
                            self.kmeans, self.index, self.total_descriptors, 
                            self.nimg_train, self._STATUS_CHECK_ITERATIONS, self.gmm)

    # start work
    loader.start()
    process.start()
    # we get the index back 
    self.index, self.train_filenames = process.join()

    print('computing database done, saving index', flush=True)
    faiss.write_index(self.index, join(self.cur_dir, 'index_full.dump'))
    self.index_full_trained = True

    print('saving filenames index', flush=True)
    pickle_dump(self.train_filenames, 'filenames_index.dump')
  
  def load_test(self):
    print('load test images', flush=True)
    total_descriptors = 0
    self.start = datetime.now()
    
    # query_database = np.zeros((self.nimg_test, self.ncentroids * self.dim))
    query_database = np.zeros((self.nimg_test, 81))
    # loop over all images
    for iteration, filename, desc in self._images_iterator(self.test_files):
      
      self.test_filenames.append(filename)
      
      # count the number of descriptors processed and print status
      total_descriptors += desc.shape[0]
      self._print_status(iteration, self.nimg_test, total_descriptors)
      
      desc = self._sanitize(desc)
      # predicted = self.kmeans.assign(desc, nclusters=1)[1]
      # query_database[iteration, :] = self._make_vlad(desc, predicted)
      query_database[iteration, :] = self._make_fisher(desc, self.gmm)
    query_database = self._sanitize(query_database)
    # print(query_database)
    return query_database

  def _make_results(self, predicted):
    results = {}
    for query_ix, query_result in enumerate(predicted):
      results[self.test_filenames[query_ix]] = ' '.join(
        [self.train_filenames[ix] for ix in query_result if ix != -1])
    return results

  def predict(self, query_database):
    print('make prediction and write submission file', flush=True)
    k = 100
    # force gpu false, index doesn't fit in gpu ram
    self.gpu = False
    if self.gpu:
      ## Search on GPU
      res = faiss.StandardGpuResources()
      gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)
      predicted = gpu_index.search(query_database, k)[1]
    else:
      # Search on CPU
      predicted = self.index.search(query_database, k)[1]
    results = self._make_results(predicted)

    submit = pd.read_csv(self.sample_submission)
    submit['images'] = submit['id'].apply(lambda ix: results.get(ix, ''))
    submit_filename = 'submit_{}_vlad_full_index.csv.gz'.format(
                datetime.now().strftime('%Y-%m-%d_%H.%M.%S'))
    submit.to_csv(submit_filename, index=False, compression='gzip')



def main():
  
  parser = argparse.ArgumentParser(description="KaggleRetrieval")
  parser.add_argument("--debug", type=bool, default=False)
  parser.add_argument("--gpu",  type=bool, default=False)
  parser.add_argument("--valid",  type=bool, default=False)
  args = parser.parse_args()

  kaggle = KaggleRetrieval(debug=args.debug, gpu=args.gpu, valid=args.valid)
  kaggle.compute_kmeans_and_index()
  kaggle.load_train()
  query_database = kaggle.load_test()
  # predict and make submit
  kaggle.predict(query_database)
  print('done', flush=True)


if __name__ == '__main__':
  main()
