
import glob
import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
from delf import feature_io
import numpy as np

def load(exemple):
  try:
    loc, _, desc, _, _ = feature_io.ReadFromFile(exemple)
  except Exception as error:
    desc = np.zeros((1, 40))
  return desc

def main():
  
  filenames = glob.glob('/home/alexandrearaujo/kaggle/landmark-retrieval-challenge/tmp/feature_test_rescale_delf/*')
  dataset = (Dataset.from_tensor_slices(filenames).apply(
      tf.contrib.data.parallel_interleave(lambda x: tf.data.Dataset.from_tensors(load(x)), cycle_length=4, block_length=16))
    )

  # create TensorFlow Iterator object
  iterator = Iterator.from_structure(dataset)
  next_element = iterator.get_next()


  # iterator = dataset.make_one_shot_iterator()
  # with tf.Session() as sess:
  #   tf.global_variables_initializer()
  #   sess.run(iterator)



if __name__ == '__main__':
  main()
