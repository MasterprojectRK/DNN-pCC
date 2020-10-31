import tensorflow as tf
import numpy as np

#compare following tutorial https://medium.com/@moritzkrger/speeding-up-keras-with-tfrecord-datasets-5464f9836c36

#parse serialized input to tensors
def parse_function(example_proto, shape):
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

#write tfRecord to disk
def writeTFRecord(pChromFactorsArray, pTargetMatricesArray, pFilename):
    with tf.io.TFRecordWriter(pFilename) as writer:
        for i in range(pChromFactorsArray.shape[0]):
            feature = {'trainFeatures':  _bytes_feature(pChromFactorsArray[i].flatten().tostring()),
                        'trainTargets':  _bytes_feature(pTargetMatricesArray[i].flatten().tostring())}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
