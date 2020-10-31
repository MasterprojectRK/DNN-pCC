import tensorflow as tf
import numpy as np
from functools import partial

#testing TFRecords and datasets
#compare following tutorial https://medium.com/@moritzkrger/speeding-up-keras-with-tfrecord-datasets-5464f9836c36

def _parse_function(example_proto, shape):
  #shape = (shape,6)
  features = {"trainFeatures": tf.io.FixedLenFeature((), tf.string),
              "trainTargets": tf.io.FixedLenFeature((), tf.string)}
  parsed_features = tf.io.parse_single_example(example_proto, features)
  feats = tf.io.decode_raw(parsed_features['trainFeatures'], tf.float64)
  targs = tf.io.decode_raw(parsed_features['trainTargets'], tf.float64)
  ws = int(shape[0] / 3)
  targ_shape = ( int(ws*(ws+1) / 2), )
  return tf.reshape(feats, shape), tf.reshape(targs, targ_shape)

# helper function from tensorflow TFRecord docs
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#similar to initial model, but smaller
def createModel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=1,
                                     kernel_size=1,
                                     activation="sigmoid",
                                     data_format="channels_last",
                                     input_shape=(10,6)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation="relu",kernel_regularizer="l2"))
    model.add(tf.keras.layers.Dense(25, activation="relu",kernel_regularizer="l2"))
    model.add(tf.keras.layers.Dense(55, activation="relu",kernel_regularizer="l2"))
    return model

def writeTFRecord(pChromFactorsArray, pTargetMatricesArray, pFilename):
    with tf.io.TFRecordWriter(pFilename) as writer:
        for i in range(pChromFactorsArray.shape[0]):
            feature = {'trainFeatures':  _bytes_feature(pChromFactorsArray[i].flatten().tostring()),
                        'trainTargets':  _bytes_feature(pTargetMatricesArray[i].flatten().tostring())}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


def main():

    nr_factors = 6
    winLength = 10
    matLength = int(winLength * (winLength+1) / 2)
    nr_batches = 100
    #the "chromatin factors"
    chromFactors = np.arange(nr_batches*winLength*nr_factors).reshape(nr_batches, winLength,nr_factors)
    #the "Hi-C submatrices", upper triangular part
    targetMatrices = np.arange(nr_batches*matLength).reshape(nr_batches,matLength)

    #to make them recognizable after loading
    chromFactors[0] = np.arange(winLength*nr_factors).reshape(winLength,nr_factors)
    targetMatrices[0] = [44]*matLength
    print(chromFactors[0])
    print(targetMatrices[0])

    filename = "test.tfrecord"
    writeTFRecord(chromFactors, targetMatrices, filename)

    shape = (winLength,nr_factors)
    ds = tf.data.TFRecordDataset(filename)
    ds = ds.map(lambda x: _parse_function(x, shape))
    ds = ds.batch(2)

    model = createModel()
    kerasOptimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=kerasOptimizer, 
                    loss=loss_fn)
    history = model.fit(ds, epochs=3, workers=8)


if __name__ == "__main__":
    main()
