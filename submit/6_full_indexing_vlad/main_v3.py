"""
Image retrieval system 

Extraction of DELF local descriptors from images [1] 
local descriptors are aggreagated with VLAD [2]
VLAD Descriptors are normalize with SSR [3]
Ajust KMeans centroids with mean of descriptors [4]
Retrieval system made with Faiss [5]

[1] Large-Scale Image Retrieval with Attentive Deep Local Features
Hyeonwoo Noh, Andre Araujo, Jack Sim, Tobias Weyand, Bohyung Han

[2] Aggregating local descriptors into a compact image representation.
H. Jégou, M. Douze, C. Schmid, and P. Pérez. 

[3] Aggregating local image descriptors into compact codes
Hervé Jégou, Florent Perronnin, Matthijs Douze, Jorge Sánchez, Patrick
Pérez, Cordelia Schmid

[4] All about VLAD 
Relja Arandjelovic. Andrew Zisserman

[5] Billion-scale similarity search with GPUs
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
from sklearn.cluster import Kmeans as sk_KMeans


kaggle_folder = '/home/alexandrearaujo/kaggle/landmark-retrieval-challenge'
storage_folder = '/media/hdd1/kaggle/landmark-retrieval-challenge/'


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


class KaggleRetrieval:

    def __init__(self, debug=False, gpu=False):

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

        self._images_iterator = self._images_iterator_pickle 

        self.nimg_train = len(self.train_files)
        self.nimg_test = len(self.test_files)

        self.train_filenames = []
        self.test_filenames = []
        self.test_ndesc_by_img = []

        # init kmeans
        self.dim = 40 # dim of descriptors
        self.ncentroids = 16
        niter = 100
        verbose = True
        self.kmeans = faiss.Kmeans(self.dim, self.ncentroids, niter, verbose)

        # init index
        self.index = faiss.IndexFlatL2(self.ncentroids * self.dim)
        
        # init index
        # nlist = 100
        # m = 8
        # quantizer = faiss.IndexFlatL2(self.ncentroids * self.dim)
        # # 8 specifies that each sub-vector is encoded as 8 bits
        # self.index = faiss.IndexIVFPQ(quantizer, self.ncentroids * self.dim, 
        #                                 nlist, m, 8)

        self.centroids_ajusted = np.zeros(self.ncentroids, self.dim)
        self.n_desc_centroids_ajusted = defaultdict(int)

        if debug:
            self.train_files = self.train_files[:5000]
            self.test_files = self.test_files[:1000]
            self.nimg_train = len(self.train_files)
            self.nimg_test = len(self.test_files)
            self._BATCH_TRAIN = 2000000
            self._STATUS_CHECK_ITERATIONS = 1000
            # self._make_results = self._make_results_debug

    def _images_iterator(self, images_path):
        for iteration, path in enumerate(images_path):
            filename = splitext(basename(path))[0]
            try:
                loc, _, desc, _, _ = feature_io.ReadFromFile(path)
            except Exception as error:
                print('problem with file {}. error : {}'.format(
                                filename, error), flush=True)
                desc = np.zeros((1, 40))
            yield iteration, filename, loc, desc

    def _images_iterator_pickle(self, images_path):
        for iteration, path in enumerate(images_path):
            filename = splitext(basename(path))[0]
            loc, desc = pickle_load(path)
            yield iteration, filename, loc, desc

    def _print_status(self, iteration, num_images, total_descriptors):
        # print status
        if iteration % self._STATUS_CHECK_ITERATIONS == 0 and iteration != 0:
            elapsed = (time.clock() - self.start)
            print('Loading image {} out of {}, last {} images took {:.5f}'
                    ' seconds, total number of descriptors {}'.format(
                        iteration, num_images, self._STATUS_CHECK_ITERATIONS, 
                        elapsed, total_descriptors), flush=True)
            self.start = time.clock()


    def _sanitize(self, data):
        """ convert array to a c-contiguous float array """
        return np.ascontiguousarray(data.astype('float32'))

    def _ajust_centroids(self, x, predicted):
        for i, (row, cluster) in enumerate(zip(x, predicted)):
            self.n_desc_centroids_ajusted[cluster] += 1
            self.centroids_ajusted[cluster] += row

    def _make_vlad(self, x, n_images, n_desc, predicted):
        
        x = x - self.kmeans.centroids[predicted]

        desc_cumsum = np.cumsum(n_desc)
        img_indexes = list(zip([0] + list(desc_cumsum[:-1]), desc_cumsum))

        database = np.zeros((n_images, self.ncentroids, self.dim))
        for ix, (start_ix, end_ix) in enumerate(img_indexes):
            x_subset = x[start_ix:end_ix]
            predicted_susbset = predicted[start_ix:end_ix]
            for cluster in range(self.ncentroids):
                mask = predicted_susbset == cluster
                if np.sum(mask):
                    database[ix, cluster, :] = x_subset[mask, :].sum(axis=0)
        database = database.reshape(n_images, self.ncentroids * self.dim)
        
        # normalize
        for ix, line in enumerate(database):
            # power normalization, also called square-rooting normalization
            line = np.sign(line) * np.sqrt(np.abs(line))
            # L2 normalization
            line = line / np.linalg.norm(line)
            database[ix, :] = line

        database = self._sanitize(database)
        return database

    def load_train_by_batch(self):
        print('______ Loading Train ______ ', flush=True)
        xtrain = []
        ndesc_by_img = []
        self.start = time.clock()
        total_descriptors = 0
        ndesc_in_batch = 0
        kmeans_trained = False
        index_trained = False

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

            # first training
            if ndesc_in_batch >= self._BATCH_TRAIN and kmeans_trained == False:
                print('total number of descriptors reached, '
                        'training kmeans', flush=True)
                
                print('training kmeans', flush=True)
                self.kmeans.train(self._sanitize(np.concatenate(xtrain)))
                kmeans_trained = True

            # add index
            if ndesc_in_batch >= self._BATCH_TRAIN and kmeans_trained == True:
                print('total number of descriptors reached, '
                        'adding descriptors to index', flush=True)
                
                xtrain = self._sanitize(np.concatenate(xtrain))
                print('kmeans on xtrain', flush=True)
                predicted = self.kmeans.index.search(xtrain, 1)[1].flatten()

                self._ajust_centroids(xtrain, predicted)
                
                n_images = len(ndesc_by_img)
                database = self._make_vlad(xtrain, n_images, 
                    ndesc_by_img, predicted)

                if not index_trained:
                    print('training index', flush=True)
                    self.index.train(database)
                    index_trained = True
                self.index.add(database)

                # reset
                ndesc_in_batch = 0
                xtrain, database, ndesc_by_img = [], [], []

        # add for last batch
        xtrain = self._sanitize(np.concatenate(xtrain))
        print('kmeans on xtrain last batch', flush=True)
        predicted = self.kmeans.index.search(xtrain, 1)[1].flatten()
        
        n_images = len(ndesc_by_img)
        database = self._make_vlad(xtrain, n_images, ndesc_by_img, predicted)
        self.index.add(database)

        # take the mean of the assign descritors to compute ajusted centroid
        for cluster, row in enumerate(self.centroids_ajusted):
            self.centroids_ajusted[cluster] = \
                row / self.n_desc_centroids_ajusted[cluster]
        # init new kmeans class with ajusted centroids
        self.ajusted_kmeans = sk_KMeans(self.centroids_ajusted, n_jobs=-1)

    def load_test(self):
        print('______ Loading Test ______ ', flush=True)
        xtest = []
        total_descriptors = 0
        self.start = time.clock()
        # loop over all images
        for iteration, filename, loc, desc in \
                self._images_iterator(self.test_files):
            
            self.test_filenames.append(filename)
            self.test_ndesc_by_img.append(desc.shape[0])
            
            # count the number of descriptors processed and print status
            total_descriptors += desc.shape[0]
            self._print_status(iteration, self.nimg_test, total_descriptors)

            # fill the database
            xtest.append(desc)

        xtest = self._sanitize(np.concatenate(xtest))

        # print('original kmeans on xtest', flush=True)
        # predicted = self.kmeans.index.search(xtest, 1)[1].flatten()

        print('centers ajusted kmeans on xtest', flush=True)
        predicted = self.ajusted_kmeans.predict(xtest).flatten()

        query_database = self._make_vlad(xtest, self.nimg_test, 
                    self.test_ndesc_by_img, predicted)
        return query_database

    def _make_results(self, predicted):
        results = {}
        for query_ix, query_result in enumerate(predicted):
            results[self.test_filenames[query_ix]] = ' '.join(
                [self.train_filenames[ix] for ix in query_result if ix != -1])
        return results

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

    def predict(self, query_database):
        print('______ Prediction ______ ', flush=True)
        k = 100
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
                                datetime.now().strftime('%Y-%m-%d.%H-%M'))
        submit.to_csv(submit_filename, index=False, compression='gzip')


def main():
    
    parser = argparse.ArgumentParser(description="KaggleRetrieval")
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--gpu",  type=bool, default=False)
    args = parser.parse_args()

    kaggle = KaggleRetrieval(debug=args.debug, gpu=args.gpu)
    kaggle.load_train_by_batch()
    query_database = kaggle.load_test()
    # predict and make submit
    kaggle.predict(query_database)

if __name__ == '__main__':
    main()