#Original file by Titu1994, changed for this project

import json
import numpy as np
import argparse
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from models import wide_residual_net as WRN, dense_net as DN
from scipy.stats import multivariate_normal

from keras.datasets import cifar10
from keras import backend as K
import keras.utils.np_utils as kutils

parser = argparse.ArgumentParser(description='CIFAR 10 Ensemble Prediction')

parser.add_argument('--optimize', type=int, default=0, help='Set to 0 to perform regular snapshot ensembles.\n'
                                                            'Set to 1 to weigh the snapshots by their accuracy on the training/validation set.\n'
                                                            'Set to 2 to weigh the snapshots based on multivariate Gaussian distributions.')

parser.add_argument('--model', type=str, default='wrn', help='Type of model to train')

parser.add_argument('--validation', action='store_true', help='Split off a part of the training data to use as validation data')

# Wide ResNet Parameters
parser.add_argument('--wrn_N', type=int, default=2, help='Number of WRN blocks. Computed as N = (n - 4) / 6.')
parser.add_argument('--wrn_k', type=int, default=4, help='Width factor of WRN')

# DenseNet Parameters
parser.add_argument('--dn_depth', type=int, default=40, help='Depth of DenseNet')
parser.add_argument('--dn_growth_rate', type=int, default=12, help='Growth rate of DenseNet')

args = parser.parse_args()

# Change to False to only predict
OPTIMIZE = args.optimize
assert OPTIMIZE in [0,1,2], 'OPTIMIZE may only have values 0, 1 and 2'

validation = args.validation

model_type = str(args.model).lower()
assert model_type in ['wrn', 'dn'], 'Model type must be one of "wrn" for Wide ResNets or "dn" for DenseNets'

files_dir = "weights"
if model_type == "wrn":
    n = args.wrn_N * 6 + 4
    k = args.wrn_k

    models_filenames = [r"%s/WRN-CIFAR10-%d-%d-1.h5" % (files_dir, n, k),
                        r"%s/WRN-CIFAR10-%d-%d-2.h5" % (files_dir, n, k),
                        r"%s/WRN-CIFAR10-%d-%d-3.h5" % (files_dir, n, k),
                        r"%s/WRN-CIFAR10-%d-%d-4.h5" % (files_dir, n, k),
                        r"%s/WRN-CIFAR10-%d-%d-5.h5" % (files_dir, n, k)]
else:
    depth = args.dn_depth
    growth_rate = args.dn_growth_rate

    models_filenames = [r"%s/DenseNet-CIFAR10-%d-%d-1.h5" % (files_dir, depth, growth_rate),
                        r"%s/DenseNet-CIFAR10-%d-%d-2.h5" % (files_dir, depth, growth_rate),
                        r"%s/DenseNet-CIFAR10-%d-%d-3.h5" % (files_dir, depth, growth_rate),
                        r"%s/DenseNet-CIFAR10-%d-%d-4.h5" % (files_dir, depth, growth_rate),
                        r"%s/DenseNet-CIFAR10-%d-%d-5.h5" % (files_dir, depth, growth_rate)]

(trainX, trainY), (testX, testY) = cifar10.load_data()
nb_classes = len(np.unique(testY))

trainX = trainX.astype('float32')
trainX /= 255.0
testX = testX.astype('float32')
testX /= 255.0

trainY_cat = kutils.to_categorical(trainY)
testY_cat = kutils.to_categorical(testY)

if (validation): # Use validation set to determine weights of the snapshots
    _, trainX, _, trainY = train_test_split(trainX, trainY, test_size=0.2, random_state=0)

if K.image_data_format() == "th":
    init = (3, 32, 32)
else:
    init = (32, 32, 3)

testX_flattened = [sample.flatten() for sample in testX]

def create_model():
    if model_type == "wrn":
        model_prefix = 'WRN-CIFAR10-%d-%d' % (args.wrn_N * 6 + 4, args.wrn_k)
        return WRN.create_wide_residual_network(init, nb_classes=10, N=args.wrn_N, k=args.wrn_k, dropout=0.00, verbose=False)
    else:
        model_prefix = 'DenseNet-CIFAR10-%d-%d' % (args.dn_depth, args.dn_growth_rate)
        return DN.create_dense_net(nb_classes=10, img_dim=init, depth=args.dn_depth, nb_dense_block=1,
                                    growth_rate=args.dn_growth_rate, nb_filter=16, dropout_rate=0.2, verbose=False)

def calculate_weighted_accuracy():
    global weighted_predictions, weight, prediction, yPred, yTrue, accuracy, error
    weighted_predictions = np.zeros((testX.shape[0], nb_classes), dtype='float32')
    for weight, prediction in zip(prediction_weights, test_preds):
        weighted_predictions += weight * prediction
    yPred = np.argmax(weighted_predictions, axis=1)
    yTrue = testY
    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    return accuracy

# Calculate train predictions of each snapshot.
train_preds = []
for fn in models_filenames:
    model = create_model()
    model.load_weights(fn)
    print("Predicting train set values on model %s" % (fn))
    yPreds = model.predict(trainX, batch_size=128, verbose=2)
    train_preds.append(yPreds)

# Calculate test predictions of each snapshot.
test_preds = []
for fn in models_filenames:
    model = create_model()
    model.load_weights(fn)
    print("Predicting test set values on model %s" % (fn))
    yPreds = model.predict(testX, batch_size=128, verbose=2)

    yPred = np.argmax(yPreds, axis=1)
    yTrue = testY
    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    print("Accuracy : ", accuracy)
    test_preds.append(yPreds)


if OPTIMIZE == 0: # Use non-weighed test predictions (standard snapshot ensembles)
    prediction_weights = [1. / len(models_filenames)] * len(models_filenames)
    accuracy = calculate_weighted_accuracy()

elif OPTIMIZE == 1: # Use weighed test predictions based on training/validation set (--validation parameter)
    training_accuracies = []
    for yPreds in train_preds:
        yPred = np.argmax(yPreds, axis=1)
        yTrue = trainY
        training_accuracies.append(metrics.accuracy_score(yTrue, yPred))

    m = min(training_accuracies)
    prediction_weights = [training_accuracy - m for training_accuracy in training_accuracies]
    accuracy = calculate_weighted_accuracy()

elif OPTIMIZE == 2: # Gaussian distributions
    # Initialize Gaussian distributions from training predictions
    densities = []
    for yPreds in train_preds:
        correct = [] # List containing all training samples this model predicted correctly
        for pred, val, trained in zip(yPreds, trainY, trainX):
            cat = np.argmax(pred)
            if cat == val:
                correct.append(trained.flatten())
        correct = np.array(correct)

        # Apply dimensionality reduction to make fitting it to a multivariate Gaussian distribution feasible
        pca = PCA(n_components=10)
        correct_reduced = pca.fit_transform(correct)
            
        # Calculate the maximum likelihood estimates of this data as a multivariate Gaussian distribution
        correct_reduced = correct_reduced[:500]
        
        mean_estimator = np.mean(correct_reduced, axis=0)
        correct_centered = correct_reduced - mean_estimator # 2D array minus 1D array --> [[1,2],[3,4]] - [5,8] = [[-4,-6],[-2,-4]]
        covariance_estimator = np.mean([np.transpose(sample[np.newaxis]) @ sample[np.newaxis] for sample in correct_centered], axis=0)
        
        distribution = multivariate_normal(mean=mean_estimator, cov=covariance_estimator)

        # Gaussian distributions created. Calculate density function on all test samples
        testX_reduced = pca.fit_transform(testX_flattened)
        cur_densities = distribution.pdf(testX_reduced)
        densities.append(cur_densities)

    # Weigh final predictions by their pdf on the different distributions
    weighted_predictions = np.zeros((testX.shape[0], nb_classes), dtype='float32')
    for cur_densities, cur_predictions in zip(densities, test_preds):
        weighted_predictions += cur_predictions * np.transpose(np.array(cur_densities)[np.newaxis])
    yPred = np.argmax(weighted_predictions, axis=1)
    yTrue = testY
    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
        
error = 100-accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)
exit()
