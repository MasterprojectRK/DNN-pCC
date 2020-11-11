import tensorflow as tf
from tensorflow.python.keras import backend
import numpy as np

#compare following tutorial https://medium.com/@moritzkrger/speeding-up-keras-with-tfrecord-datasets-5464f9836c36

#parse serialized input to tensors
def parse_function(example_proto, shapeDict):
    feats = None
    targs = None
    dna = None
    features = {"features": tf.io.FixedLenFeature((), tf.string),
                "targets": tf.io.FixedLenFeature((), tf.string),
                "dna": tf.io.FixedLenFeature((), tf.string)}
    parsed_features = tf.io.parse_single_example(example_proto, features)
    if "feats" in shapeDict:
        feats = tf.io.decode_raw(parsed_features['features'], tf.float64)
        feats = tf.reshape(feats, shapeDict["feats"])
    if "targs" in shapeDict:
        targs = tf.io.decode_raw(parsed_features['targets'], tf.float64)
        targs = tf.reshape(targs, shapeDict["targs"])
    if "dna" in shapeDict:
        dna = tf.io.decode_raw(parsed_features['dna'], tf.uint8)
        dna = tf.reshape(dna, shapeDict["dna"])
    retList = []
    featDict = dict()
    if feats is not None:
        featDict["feats"] = feats #chromatin features
    if dna is not None:
        featDict["dna"] = dna #dna sequence
    if len(featDict) > 0:
        retList.append(featDict)
    if targs is not None:
        retList.append(targs)   #target submatrices
    return tuple(retList)

# helper functions from tensorflow TFRecord docs
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#write tfRecord to disk
def writeTFRecord(pFilename, pRecordDict):
    if not isinstance(pFilename, str):
        return
    if not isinstance(pRecordDict, dict):
        return
    for key in pRecordDict:
        if not isinstance(pRecordDict[key], np.ndarray):
            return
    batches = set()
    for key in pRecordDict:
        batches.add(pRecordDict[key].shape[0])
    if len(batches) > 1:
        msg = "Batch sizes are not equal"
        raise ValueError(msg)
    
    with tf.io.TFRecordWriter(pFilename, options="GZIP") as writer:
        for i in range(list(batches)[0]):
            feature = dict()
            for key in pRecordDict:
                feature[key] = _bytes_feature( pRecordDict[key][i].flatten().tostring() )
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
