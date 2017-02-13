from __future__ import division, print_function, absolute_import
import tflearn
from skimage.feature import canny
from sklearn.base import TransformerMixin
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from sklearn.cross_validation import train_test_split
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import sklearn
import csv
from skimage.filters import sobel
import tensorflow as tf
from sklearn.svm import LinearSVC
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score
n_classes=2
x = tf.placeholder(tf.float32, [None,143,4,1])
y= tf.placeholder(tf.float32, [None,n_classes])
imList=[]
NUM_CLASSES = 2 
IMAGE_WIDTH = 176
IMAGE_HEIGHT=208
CHANNELS = 1
IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS
def imhist(im):
  # calculates normalized histogram of an image
	m, n = im.shape
	h = [0.0] * 256
	for i in range(m):
		for j in range(n):
			h[im[i, j]]+=1
	return np.array(h)/(m*n)

def cumsum(h):
	# finds cumulative sum of a numpy array, list
	return [sum(h[:i+1]) for i in range(len(h))]

def histeq(im):
	#calculate Histogram
	h = imhist(im)
	cdf = np.array(cumsum(h)) #cumulative distribution function
	sk = np.uint8(255 * cdf) #finding transfer function values
	s1, s2 = im.shape
	Y = np.zeros_like(im)
	# applying transfered values for each pixels
	for i in range(0, s1):
		for j in range(0, s2):
			Y[i, j] = sk[im[i, j]]
	H = imhist(Y)
	#return transformed image, original and new istogram, 
	# and transform function
	return Y , h, H, sk
def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr
for root, dirs, files in os.walk("/Users/elhamalkabawi/Desktop/disc2"):
    for file in files:
        if file.endswith(".gif"):
            p = os.path.join(root, file)
            img = Image.open(p, 'r')
            img= img.resize((IMAGE_WIDTH,IMAGE_HEIGHT))
            img=np.array(img)
            new_img, h, new_h, sk = histeq(img)
            elevation_map = sobel(new_img)
            edges = canny(elevation_map /255.)
            imge=normalize(edges)
            imList.append(imge)
imList = np.asarray(img, dtype='float32') / 256.
X_train, X_test = train_test_split(imList, test_size = 0.4)  
print(len(X_train))
X_train = np.array(X_train)
print(X_train.shape)
acc=tflearn.metrics.Accuracy()
#X_train = X_train.reshape([62,44,8,1])
#y_train=y_train.reshape([4576 ,2])
X_test=np.array(X_test)
n_input=176
#X_test=X_test.reshape([42,44,8,1])
#y_test=y_test.reshape([4576 ,2])
label=[]
n_hidden_1 = 100 # 1st layer num features
n_hidden_2 = 50 # 2nd layer num features
import pandas
from sklearn.metrics import classification_report
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
names=['map','count','Age','eTIV','ETA','nWBV','L. HC','R. HC','CDR']
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

csv = pd.read_csv('/Users/elhamalkabawi/Desktop/research/OASIS_data.csv',names=names)
#meanDia = np.mean
xt= DataFrameImputer().fit_transform(csv)
array = xt.values

#X_train,y_train=np.split(train,2)
#X_test,y_test=np.split(test,2)
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
#array = csv.values
X = array[1:737,0:8]
Y = array[1:737,8]
label=[]
for i in Y:
    if i=='No dementia':
            label.append(0.0)
    else:
        if i=='Incipient demt PTP':
            #print i
            label.append(0.5)
        else:
           if i=='uncertain dementia':
               label.append(1.0)
           else:
                   if i=='DAT':
                           label.append(2.0)
                   else:
                           label.append(3.0)
y_train, y_test= sklearn.cross_validation.train_test_split(label, train_size = 0.772)
y_train=np.array(y_train)
y_test=np.array(y_test)
print(y_train.shape)
print(y_test.shape)

y_train=y_train.reshape([142,4])
y_test=y_test.reshape([42,4])
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

encoder_op = encoder(X_train)
print(encoder_op )
decoder_op = decoder(encoder_op)
print(encoder_op )
# Prediction
y_pred = decoder_op
print('pred')
print(y_pred)

# Targets (Labels) are the input data.
y_true = X_train
print(y_true)
encoder_op1 = encoder(X_test)
print(encoder_op1 )
decoder_opn = decoder(encoder_op1)

# Prediction
pred = decoder_opn
print(pred)
init = tf.initialize_all_variables()
#########################
with tf.Session() as sess:
        sess.run(init)
        y_pred=y_pred.eval()
        predicted=y_pred.reshape([62,44,8,1])
        pred=pred.eval()
        pred=pred.reshape([42,44,8,1])
network = input_data(shape=[None,44,8,1], name='input')
network = conv_2d(network, 16, 8, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
print(network.get_shape())
network = conv_2d(network, 32, 4, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
print(network.get_shape())
network = conv_2d(network, 64, 2, activation='relu', regularizer="L2")
network = max_pool_2d(network, 1)
network = local_response_normalization(network)
network = conv_2d(network, 128,2, activation='relu', regularizer="L2")
network = max_pool_2d(network, 1)
network = local_response_normalization(network)
print(network.get_shape())
network = conv_2d(network, 256, 1, activation='relu', regularizer="L2")
network = max_pool_2d(network, 1)
network = local_response_normalization(network)
network = conv_2d(network, 1024,1, activation='relu', regularizer="L2")
network = max_pool_2d(network, 1)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 4, activation='softmax')

#network= LinearSVC(network, loss='squared_hinge', penalty='l2',multi_class='ovr')
network = regression(network, optimizer='sgd', learning_rate=0.01,metric=acc,
                     loss='categorical_crossentropy', name='target')
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': predicted}, {'target': y_train}, n_epoch=100,
          validation_set=({'input': pred}, {'target': y_test}))
#print(X_test.shape)

#n=model.predict(X_test)
#print(n)
result = model.evaluate(pred,y_test)
#print(result)
result2=model.evaluate(predicted,y_train)
accuracy_score1 = result2[0]
accuracy_score = result[0]
print('\n\ntesting accuracy: {:0.04%}\n\n'.format(accuracy_score))
print('\n\ntraining accuracy: {:0.04%}\n\n'.format(accuracy_score1))
#print(network)
preds=model.predict(pred)
print(y_test)
preds=np.array(preds)
preds=preds.reshape([42,4])
print(preds)
#matrix = confusion_matrix(y_test, pred)
#score = accuracy_score(y_train, pred)
#print("Accuracy: %f" % score)
#print(X_train)
predictions = tf.cast(tf.greater(network, 4), tf.int64)
y= tf.placeholder(shape=[None, 4], dtype=tf.int64, name="Y")
print ("predictions=%s" % predictions)
#preds=model.predict(pred)
#print(classification_report(Y, predictions))
#binary_accuracy_op(c, y_)
print(preds)

Ybool = tf.cast(y, tf.bool)
print ("Ybool=%s" % Ybool)

pos = tf.boolean_mask(predictions, Ybool)
print(pos)
neg = tf.boolean_mask(predictions, ~Ybool)
psize = tf.cast(tf.shape(pos)[0], tf.int64)
print(psize)
nsize = tf.cast(tf.shape(neg)[0], tf.int64)
true_positive = tf.reduce_sum(pos, name="true_positive")
false_negative = tf.sub(psize, true_positive, name="false_negative")
false_positive = tf.reduce_sum(neg, name="false_positive")
true_negative = tf.sub(nsize, false_positive, name="true_negative")
overall_accuracy = tf.truediv(tf.add(true_positive, true_negative), tf.add(nsize, psize), name="overall_accuracy")
vmset = [true_positive, true_negative, false_positive, false_negative, overall_accuracy]
print(vmset)
