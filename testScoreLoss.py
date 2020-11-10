import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation
from testScoreModel import CustomReshapeLayer, DiamondLayer, buildMatrixBranch, makeTrainTestSamples

np.random.seed(35)

def main_fn():
    #params
    matsize = 10
    diamondsize = 2
    nr_factors = 6
    nr_batches = 10
    learningrate = 1e-5
    nr_epochs = 10
    modelPlotName = "scoreLoss.png"

    train_features, targ_matrices, targ_scores = makeTrainTestSamples(nr_batches=nr_batches, matsize=matsize, nr_factors=nr_factors, diamondsize=diamondsize)

    model = buildModel(matsize,nr_factors, nr_batches)
    model.summary()
    tf.keras.utils.plot_model(model,show_shapes=True, to_file=modelPlotName)  

    opt = tf.keras.optimizers.Adam(learning_rate=learningrate, decay=learningrate/nr_epochs)
    losses = {
        "matrix_out": tf.keras.losses.MeanSquaredError(),
        "score_out": customLossWrapper(pMatrixsize=matsize, pDiamondsize=diamondsize, pBatchsize=nr_batches)
    }
    lossWeights = {"matrix_out": 1.0, "score_out": 1.0}
    model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights)

    model.fit(x=train_features, 
            y={"matrix_out": targ_matrices, "score_out": targ_scores},
            epochs=nr_epochs,
            verbose=1,
    )


def buildModel(pMatsize, pNrFactors, pBatchsize):
    inputs = Input(shape=(pMatsize,pNrFactors))
    matBranch = buildMatrixBranch(inputs, pMatsize, pNrFactors)
    dummyBranch = Activation("linear", name="score_out")(matBranch) #pass-through
    model = Model(inputs=inputs, outputs=[matBranch, dummyBranch], name="testScoreLoss")
    return model

def customLossWrapper(pMatrixsize, pDiamondsize, pBatchsize):
    def customLoss(y_true, y_pred):
        #compute the score from the predicted (flattened) upper triangular matrix
        predScore = CustomReshapeLayer(pMatrixsize,pBatchsize)(y_pred)
        predScore = DiamondLayer(pMatrixsize,pDiamondsize,pBatchsize)(predScore)
        #compute mean squared error
        predLoss = tf.square(y_true - predScore)
        predLoss = tf.reduce_mean(predLoss)
        return predLoss
    return customLoss

if __name__ == "__main__":
    main_fn()