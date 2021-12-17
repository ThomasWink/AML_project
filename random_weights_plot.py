#Original file of this project

import numpy as np
import argparse
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from models import wide_residual_net as WRN, dense_net as DN
from keras.datasets import cifar10, cifar100
from keras import backend as K
import keras.utils.np_utils as kutils

parser = argparse.ArgumentParser(description='Plot accuracy distribution when weighing the snapshots randomly')

# Wide ResNet Parameters
parser.add_argument('--wrn_N', type=int, default=2, help='Number of WRN blocks. Computed as N = (n - 4) / 6.')
parser.add_argument('--wrn_k', type=int, default=4, help='Width factor of WRN')

# DenseNet Parameters
parser.add_argument('--dn_depth', type=int, default=40, help='Depth of DenseNet')
parser.add_argument('--dn_growth_rate', type=int, default=12, help='Growth rate of DenseNet')

args = parser.parse_args()

np.random.seed(0)

n = args.wrn_N * 6 + 4
k = args.wrn_k
depth = args.dn_depth
growth_rate = args.dn_growth_rate

files_dir = "weights"
models_filenames = {'cifar10': {}, 'cifar100': {}}
models_filenames['cifar10']['wrn'] = [r"%s/WRN-CIFAR10-%d-%d-1.h5" % (files_dir, n, k),
                        r"%s/WRN-CIFAR10-%d-%d-2.h5" % (files_dir, n, k),
                        r"%s/WRN-CIFAR10-%d-%d-3.h5" % (files_dir, n, k),
                        r"%s/WRN-CIFAR10-%d-%d-4.h5" % (files_dir, n, k),
                        r"%s/WRN-CIFAR10-%d-%d-5.h5" % (files_dir, n, k)]
models_filenames['cifar100']['wrn'] = [r"%s/WRN-CIFAR100-%d-%d-1.h5" % (files_dir, n, k),
                        r"%s/WRN-CIFAR100-%d-%d-2.h5" % (files_dir, n, k),
                        r"%s/WRN-CIFAR100-%d-%d-3.h5" % (files_dir, n, k),
                        r"%s/WRN-CIFAR100-%d-%d-4.h5" % (files_dir, n, k),
                        r"%s/WRN-CIFAR100-%d-%d-5.h5" % (files_dir, n, k)]
models_filenames['cifar10']['dn'] = [r"%s/DenseNet-CIFAR10-%d-%d-1.h5" % (files_dir, depth, growth_rate),
                        r"%s/DenseNet-CIFAR10-%d-%d-2.h5" % (files_dir, depth, growth_rate),
                        r"%s/DenseNet-CIFAR10-%d-%d-3.h5" % (files_dir, depth, growth_rate),
                        r"%s/DenseNet-CIFAR10-%d-%d-4.h5" % (files_dir, depth, growth_rate),
                        r"%s/DenseNet-CIFAR10-%d-%d-5.h5" % (files_dir, depth, growth_rate)]
models_filenames['cifar100']['dn'] = [r"%s/DenseNet-CIFAR100-%d-%d-1.h5" % (files_dir, depth, growth_rate),
                        r"%s/DenseNet-CIFAR100-%d-%d-2.h5" % (files_dir, depth, growth_rate),
                        r"%s/DenseNet-CIFAR100-%d-%d-3.h5" % (files_dir, depth, growth_rate),
                        r"%s/DenseNet-CIFAR100-%d-%d-4.h5" % (files_dir, depth, growth_rate),
                        r"%s/DenseNet-CIFAR100-%d-%d-5.h5" % (files_dir, depth, growth_rate)]

def create_model():
    if model_type == "wrn":
        model_prefix = 'WRN-%d-%d' % (args.wrn_N * 6 + 4, args.wrn_k)
        return WRN.create_wide_residual_network(init, nb_classes=nb_classes, N=args.wrn_N, k=args.wrn_k, dropout=0.00, verbose=False)
    else:
        model_prefix = 'DenseNet-%d-%d' % (args.dn_depth, args.dn_growth_rate)
        return DN.create_dense_net(nb_classes=nb_classes, img_dim=init, depth=args.dn_depth, nb_dense_block=1,
                                    growth_rate=args.dn_growth_rate, nb_filter=16, dropout_rate=0.2, verbose=False)

def calculate_weighted_accuracy():
    weighted_predictions = np.zeros((testX.shape[0], nb_classes), dtype='float32')
    for weight, prediction in zip(prediction_weights, test_preds):
        weighted_predictions += weight * prediction
    yPred = np.argmax(weighted_predictions, axis=1)
    yTrue = testY
    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    return accuracy

pa = []
for dataset in ['cifar10', 'cifar100']:
    if dataset == 'cifar10':
        (trainX, trainY), (testX, testY) = cifar10.load_data()
    else:
        (trainX, trainY), (testX, testY) = cifar100.load_data()
        
    nb_classes = len(np.unique(testY))

    trainX = trainX.astype('float32')
    trainX /= 255.0
    testX = testX.astype('float32')
    testX /= 255.0

    trainY_cat = kutils.to_categorical(trainY)
    testY_cat = kutils.to_categorical(testY)

    if K.image_data_format() == "th":
        init = (3, 32, 32)
    else:
        init = (32, 32, 3)
        
    for model_type in ['wrn', 'dn']:
        # Calculate test predictions of each snapshot.
        test_preds = []
        for fn in models_filenames[dataset][model_type]:
            model = create_model()
            model.load_weights(fn)
            print("Predicting test set values on model %s" % (fn))
            yPreds = model.predict(testX, batch_size=128, verbose=2)
            test_preds.append(yPreds)


        # Calculate unweighed snapshot ensemble accuracy
        prediction_weights = [1 for _ in range(len(test_preds))]
        unweighed_accuracy = calculate_weighted_accuracy()

        # Calculate ensemble accuracy with 10000 different random weights
        accuracy_count = {}
        for i in range(1000):
            prediction_weights = np.random.random(len(test_preds))
            accuracy = calculate_weighted_accuracy()
            '''weighted_predictions = np.zeros((testX.shape[0], nb_classes), dtype='float32')
            for weight, prediction in zip(prediction_weights, test_preds):
                weighted_predictions += weight * prediction
            yPred = np.argmax(weighted_predictions, axis=1)
            yTrue = testY
            accuracy = metrics.accuracy_score(yTrue, yPred) * 100'''
            if accuracy not in accuracy_count:
                accuracy_count[accuracy] = 0
            accuracy_count[accuracy] += 1

        print(unweighed_accuracy, max(accuracy_count.keys()))
        sort = sorted(accuracy_count.keys())
        p, = plt.plot(sort, [accuracy_count[acc] for acc in sort])
        pa.append(p)
        plt.axvline(unweighed_accuracy, 0, 1, linestyle='--', color=p.get_color())

plt.xlabel("Accuracy")
plt.ylabel("Number of configurations")
plt.legend(pa, ['Cifar 10 WRN', 'Cifar 10 DenseNet', 'Cifar 100 WRN', 'Cifar 100 DenseNet'])
plt.xticks(np.arange(35,95,2))
plt.rc('axes', titlesize=50)
plt.rc('axes', labelsize=50)
plt.savefig('tst.pdf', dpi=400, format='pdf')
plt.show()
