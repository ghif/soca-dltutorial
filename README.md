# soca-dltutorial

Kode Logistic Regression dan Feedforward Neural Nets dalam python yang digunakan pada sesi tutorial SOCA tentang deep learning.

Dependencies:
* python >= 2.7.x
* Numpy / Scipy
* Theano: http://deeplearning.net/software/theano/
* Keras: http://keras.io/ 



## Main file
* logistic_regression.py
* neural_net.py


### Dataset
* usps.pkl.gz: file yang berisi handwritten digits dalam format pickle, diadopsi dari https://www.otexts.org/1577

Dapat di-load dengan cara sbb:
```python
import cPickle as pickle
import gzip
(X_train, Y_train), (X_test, Y_test) = pickle.load(gzip.open("usps.pkl.gz","rb"))
```

### IPython
Jika anda menggunakan (IPython Notebook/Jupyter)[https://jupyter.org/] , simulasi dapat dijalankan dari web browser dengan cara:
```python
ipython notebook soca_demo.ipynb
```