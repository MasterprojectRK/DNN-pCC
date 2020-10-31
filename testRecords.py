import tensorflow as tf
import numpy as np

#testing TFRecords and datasets
#compare following tutorial https://medium.com/@moritzkrger/speeding-up-keras-with-tfrecord-datasets-5464f9836c36

def _parse_function(example_proto):
  features = {"trainFeatures": tf.io.FixedLenFeature((), tf.string),
              "trainTargets": tf.io.FixedLenFeature((), tf.string)}
  parsed_features = tf.io.parse_single_example(example_proto, features)
  feats = tf.io.decode_raw(parsed_features['trainFeatures'], tf.int64)
  targs = tf.io.decode_raw(parsed_features['trainTargets'], tf.int64)
  return tf.reshape(feats, (10,6)), tf.reshape(targs, (55,))


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


#the "chromatin factors"
chromFactors = np.arange(100*10*6).reshape(100,10,6)
#the "Hi-C submatrices", upper triangular part
targetMatrices = np.arange(100*55).reshape(100,55)

#to make them recognizable after loading
chromFactors[0] = np.arange(60).reshape(10,6)
targetMatrices[0] = [44]*55
print(chromFactors[0])
print(targetMatrices[0])


with tf.io.TFRecordWriter("test.tfrecord") as writer:
    for i in range(chromFactors.shape[0]):
        feature = {'trainFeatures':  _bytes_feature(chromFactors[i].flatten().tostring()),
                    'trainTargets':  _bytes_feature(targetMatrices[i].flatten().tostring())}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())


ds = tf.data.TFRecordDataset("test.tfrecord")
ds = ds.map(_parse_function)
ds = ds.batch(2)

model = createModel()
kerasOptimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = tf.keras.losses.MeanSquaredError()
model.compile(optimizer=kerasOptimizer, 
                 loss=loss_fn)
history = model.fit(ds, epochs=3, workers=8)



