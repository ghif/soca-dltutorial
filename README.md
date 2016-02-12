# soca-dltutorial

Implementasi Logistic Regression dan Feedforward Neural Nets dalam python.

## Main file
* logistic_regression.py
* neural_net.py


### Dataset
* usps.pkl.gz: file yang berisi handwritten digits

Dapat di-load dengan cara sbb:
> import cPickle as pickle

> import gzip

> (X_train, Y_train), (X_test, Y_test) = pickle.load(gzip.open("usps.pkl.gz","rb"))