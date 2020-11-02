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
def writeTFRecord(pChromFactorsArray, pDNASequenceArray, pTargetMatricesArray, pFilename):
    batches = set()
    if isinstance(pChromFactorsArray, np.ndarray) \
            and not len(pChromFactorsArray.shape) == 3:
        msg = "Chrom factors array wrong shape"
        print(msg)
        return
    elif isinstance(pChromFactorsArray, np.ndarray):
        batches.add( pChromFactorsArray.shape[0] )
    if isinstance(pDNASequenceArray, np.ndarray) \
                and not len(pDNASequenceArray.shape) == 3:
        msg = "DNA seq. array wrong shape"
        print(msg)
        return
    elif isinstance(pDNASequenceArray, np.ndarray):
        batches.add( pDNASequenceArray.shape[0] )
    if isinstance(pTargetMatricesArray, np.ndarray) \
                and not len(pTargetMatricesArray.shape) == 2:
        msg = "target matrices array wrong shape"
        print(msg)
        return
    elif isinstance(pTargetMatricesArray, np.ndarray):
        batches.add( pTargetMatricesArray.shape[0] )
    
    if len(batches) != 1:
        msg = "no. of batches wrong or no data"
        print(msg)
        return
    
    if not isinstance(pFilename, str):
        msg = "Filename must be a string"
        print(msg)
        return
    if not isinstance(pChromFactorsArray, np.ndarray) \
            and not isinstance(pDNASequenceArray, np.ndarray) \
            and not isinstance(pTargetMatricesArray, np.ndarray):
        msg = "Nothing to write"
        print(msg)
        return
    
    with tf.io.TFRecordWriter(pFilename) as writer:
        for i in range(list(batches)[0]):
            feature = dict()
            if isinstance(pChromFactorsArray, np.ndarray):
                feature['features'] =  _bytes_feature( pChromFactorsArray[i].flatten().tostring() ),
            else:
                feature['features'] = _bytes_feature( np.array([0]).tostring() )
            if isinstance(pTargetMatricesArray, np.ndarray):
                feature['targets'] =  _bytes_feature( pTargetMatricesArray[i].flatten().tostring() )
            else:
                feature['targets'] = _bytes_feature( np.array([0]).tostring() )
            if isinstance(pDNASequenceArray, np.ndarray):
                feature['dna'] = _bytes_feature( pDNASequenceArray[i].flatten().tostring() )
            else:
                feature['dna'] = _bytes_feature( np.array([0]).tostring() )
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
