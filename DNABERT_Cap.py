import os
import pickle
import keras
from keras.initializers import GlorotUniform, Zeros
import pandas as pd
data = pd.read_csv("Data_Hela_random.csv")

import torch
from transformers import AutoTokenizer, BertModel
from tqdm import tqdm as tqdm

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6",trust_remote_code=True)model = BertModel.from_pretrained("zhihan1996/DNA_bert_6")

y=data['label'].values
from sklearn.preprocessing import LabelEncoder
import tensorflow
from tensorflow.keras.utils import to_categorical
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
dummy_y = to_categorical(encoded_Y)

tokenized =data['sequence'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))


import numpy as np
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
attention_mask = np.where(padded != 0, 1, 0)


input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)
model.to(device)

input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

features = last_hidden_states[0]

#@title
import keras.backend as K
import tensorflow as tf
from keras import initializers, layers

class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), -1) + K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, d1, d2] by the max value in axis=1.
    Output shape: [None, d2]
    """

    def call(self, inputs, **kwargs):
        # use true label to select target capsule, shape=[batch_size, num_capsule]
        if type(inputs) is list:  # true label is provided with shape = [batch_size, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of vectors of capsules
            #x = inputs
            # Enlarge the range of values in x to make max(new_x)=1 and others < 0
            #x = (x - K.max(x, 1, True)) / K.epsilon() + 1
            #mask = K.clip(x, 0, 1)  # the max value in x clipped to 1 and other to 0

        # masked inputs, shape = [batch_size, dim_vector]
        #inputs_masked = tf.matmul(inputs, mask)
          x = tf.sqrt(tf.reduce_sum(tf.square(inputs), -1))
          mask = tf.one_hot(indices=tf.argmax(x, 1), depth=x.shape[1])
          masked = K.batch_flatten(inputs * tf.expand_dims(mask, -1))
        return masked
        #return inputs_masked


    #def compute_output_shape(self, input_shape):
        #if type(input_shape[0]) is tuple:  # true label provided
         #   return tuple([None, input_shape[0][-1]])
        #else:
         #   return tuple([None, input_shape[-1]])

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config
def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_vector] and output shape = \
    [None, num_capsule, dim_vector]. For Dense Layer, input_dim_vector = dim_vector = 1.

    :param num_capsule: number of capsules in this layer
    :param dim_vector: dimension of the output vectors of the capsules in this layer
    :param num_routings: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_vector, num_routing=3,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_vector = dim_vector
        self.num_routing = num_routing
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_capsule': self.num_capsule,
            'dim_vector': self.dim_vector,
            'num_routing': self.num_routing,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
        })
        return config

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_vector]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_vector = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule, self.dim_vector,self.input_dim_vector],
                                 initializer=self.kernel_initializer,
                                 name='W')

        # Coupling coefficient. The redundant dimensions are just to facilitate subsequent matrix calculation.
        #self.bias = self.add_weight(shape=[1, self.input_num_capsule, self.num_capsule, 1, 1],
                                    #initializer=self.bias_initializer,
                                    #name='bias',
                                    #trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        print('This is input:', inputs.shape)
        # inputs.shape=[None, input_num_capsule, input_dim_vector]
        # Expand dims to [None, input_num_capsule, 1, 1, input_dim_vector]
        inputs_expand = tf.expand_dims(tf.expand_dims(inputs, 1), -1)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # Now it has shape = [None, input_num_capsule, num_capsule, 1, input_dim_vector]
        inputs_tiled = tf.tile(inputs_expand, [1, self.num_capsule, 1, 1, 1])

        #inputs_tiled  = tf.expand_dims(inputs_tiled, 4)
        print('This is inputs_expand:', inputs_expand.shape)
        print('This is inputs_tiled:', inputs_tiled.shape)

        print('This is W:', self.W.shape)



        """
        # Compute `inputs * W` by expanding the first dim of W. More time-consuming and need batch_size.
        # Now W has shape  = [batch_size, input_num_capsule, num_capsule, input_dim_vector, dim_vector]
        w_tiled = K.tile(K.expand_dims(self.W, 0), [self.batch_size, 1, 1, 1, 1])

        # Transformed vectors, inputs_hat.shape = [None, input_num_capsule, num_capsule, 1, dim_vector]
        inputs_hat = K.batch_dot(inputs_tiled, w_tiled, [4, 3])
        """
        # Compute `inputs * W` by scanning inputs_tiled on dimension 0. This is faster but requires Tensorflow.
        # inputs_hat.shape = [None, input_num_capsule, num_capsule, 1, dim_vector]
        inputs_hat = tf.squeeze(tf.map_fn(lambda x: tf.matmul(self.W, x), elems=inputs_tiled))
        #b=tf.zeros(shape=[inputs.shape[0] ,self.num_capsule,1,self.input_num_capsule])
        b = tf.zeros(shape=[tf.shape(inputs)[0], self.num_capsule, 1, self.input_num_capsule])


        #inputs_hat = tf.map_fn(lambda x: tf.matmul(x,self.W), elems=inputs_tiled)
        print('This is inputs_hat (`inputs * W`):', inputs_hat.shape)

        """
        # Routing algorithm V1. Use tf.while_loop in a dynamic way.
        def body(i, b, outputs):
            c = tf.nn.softmax(self.bias, dim=2)  # dim=2 is the num_capsule dimension
            outputs = squash(K.sum(c * inputs_hat, 1, keepdims=True))
            b = b + K.sum(inputs_hat * outputs, -1, keepdims=True)
            return [i-1, b, outputs]

        cond = lambda i, b, inputs_hat: i > 0
        loop_vars = [K.constant(self.num_routing), self.bias, K.sum(inputs_hat, 1, keepdims=True)]
        _, _, outputs = tf.while_loop(cond, body, loop_vars)
        """
        # Routing algorithm V2. Use iteration. V2 and V1 both work without much difference on performance
        assert self.num_routing > 0, 'The num_routing should be > 0.'
        for i in range(self.num_routing):

            #c = tf.nn.softmax(b, axis=1)
            c=tf.transpose(tf.nn.softmax(tf.transpose(b,perm=[0,2,3,1])),perm=[0,3,1,2])
            print("c is:",c)
            # outputs.shape=[None, 1, num_capsule, 1, dim_vector]
            outputs = squash(tf.matmul(c, inputs_hat))
            print("Output is:",outputs.shape)

            # last iteration needs not compute bias which will not be passed to the graph any more anyway.
            if i < self.num_routing - 1:
                # self.bias = K.update_add(self.bias, K.sum(inputs_hat * outputs, [0, -1], keepdims=True))
                b += tf.matmul(outputs, inputs_hat, transpose_b=True)            # tf.summary.histogram('BigBee', self.bias)  # for debugging
        return tf.squeeze(outputs)


    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_vector])




def PrimaryCap(inputs, dim_vector, n_channels, kernel_size, strides, padding):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_vector: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_vector]
    """
    output = layers.Conv1D(filters=dim_vector*n_channels, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_vector])(output)
    print("Outputs PrimaryCap:",outputs.shape)
    return layers.Lambda(squash)(outputs)

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    # return tf.reduce_mean(tf.square(y_pred))
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))

    Varmean = tf.reduce_mean(tf.reduce_sum(L, 1))
    return Varmean

from keras import layers, models
dropout_p = 0.3
from keras import backend as K
from tensorflow.keras.utils import to_categorical


def CapsNet(input_shape, n_class, num_routing):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 4d, [None, width, height, channels]
    :param n_class: number of classes
    :param num_routing: number of routing iterations
    :return: A Keras Model with 2 inputs and 2 outputs
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv1D layer
    conv1 = layers.Conv1D(filters=64, kernel_size=2, strides=1, padding='valid', activation='relu', name='conv1')(x)
    lstm = LSTM(64, dropout=dropout_p, recurrent_dropout=dropout_p,return_sequences=True)(conv1)
    # Layer 2: Conv1D layer with `squash` activation, then reshape to [None, num_capsule, dim_vector]
    primarycaps = PrimaryCap(lstm, dim_vector=8, n_channels=8, kernel_size=2, strides=2, padding='valid')

    #print("PrimaryCaps:",primarycaps.shape)
    #Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='digitcaps')(primarycaps)
    #print("DigitCaps:",digitcaps.shape)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)

    out_caps = Length(name='out_caps')(digitcaps)
    # two-input-two-output keras Model
    return models.Model([x], [out_caps])

from keras.layers import Activation, Add, Bidirectional, Conv1D, Dense, Dropout, Embedding, Flatten
from keras.layers import concatenate, GRU, Input, LSTM, MaxPooling1D
from keras.layers import GlobalAveragePooling1D,  GlobalMaxPooling1D, SpatialDropout1D
from keras.models import Model

model = CapsNet(input_shape=(98,768),
                n_class=2, num_routing =3)
model.summary()

model.compile(optimizer='adam',loss=margin_loss, metrics=[tf.keras.metrics.BinaryAccuracy()])



import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow
from tensorflow.keras.utils import to_categorical
import re
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn import metrics
import numpy as np
import os
import pickle
from imblearn.metrics import specificity_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot

final_ft = []
acc = []
prec = []
rec = []
f1 = []
auc = []
sp = []
mcc = []
roc_auc = []

from sklearn.model_selection import train_test_split

for jj in range(0,20):
    X_train, X_test, y_train, y_test = train_test_split(features,dummy_y, test_size = 0.20, random_state = 0)

    X_train = X_train.data.cpu().numpy() 
    y_train = np.array(y_train)
    X_test = X_test.data.cpu().numpy()
    y_test = np.array(y_test)

    model.fit(X_train, y_train, batch_size=64,epochs=12,validation_split=0.10,verbose=1) 

    y_pred =  model.predict(X_test)
    test_labels = y_test
    test_prediction = np.argmax(y_pred, axis=-1)
    test_true = np.argmax(test_labels, axis=-1)
    fpr, tpr, thresholds = metrics.roc_curve(test_true, test_prediction, pos_label=1)
    print("Accuracy is:",accuracy_score(test_true,test_prediction)*100)
    print("Recall is:",recall_score(test_true,test_prediction))
    print("Specificity is:",specificity_score(test_true,test_prediction, average='weighted'))
    print("MCC is:",matthews_corrcoef(test_true,test_prediction))
    print("ROC-AUC is:",roc_auc_score(test_true,y_pred[:, 1]))
    print("Precision is:",precision_score(test_true,test_prediction))
    print("F1-Score is:",f1_score(test_true,test_prediction))
    lr_probs = y_pred[:, 1]
    ns_probs = [0 for _ in range(len(y_test))]
    ns_auc = roc_auc_score(test_true, ns_probs)
    lr_auc = roc_auc_score(test_true, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Proposed: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(test_true, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(test_true, lr_probs)
    print(ns_fpr)
    print(ns_tpr)
    print(lr_fpr)
    print(lr_tpr)
    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Proposed')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()



    acc.append(accuracy_score(test_true,test_prediction)*100)
    prec.append(precision_score(test_true,test_prediction))
    rec.append(recall_score(test_true,test_prediction))
    sp.append(specificity_score(test_true,test_prediction, average='weighted'))
    f1.append(f1_score(test_true,test_prediction))
    auc.append(metrics.auc(fpr, tpr))
    mcc.append(matthews_corrcoef(test_true,test_prediction))
    roc_auc.append(roc_auc_score(test_true,y_pred[:, 1]))


print("Mean Accuracy is:",np.mean(acc))
print("Mean Precision is:",np.mean(prec))
print("Mean Recall is:",np.mean(rec))
print("Mean Specificity is:",np.mean(sp))
print("Mean F1-Score is:",np.mean(f1))
print("Mean AUC is:",np.mean(auc))
print("Mean MCC is:",np.mean(mcc))
print("Mean ROC-AUC is:",np.mean(roc_auc))



