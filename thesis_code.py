# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 06:35:49 2021

@author: -
"""

from keras.preprocessing import text
from keras.utils import np_utils
from keras.preprocessing import sequence
import numpy as np

import pandas as pd
import nltk
import gensim
from gensim import corpora, models, similarities


import matplotlib.pyplot as plt
from collections import Counter

from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import itertools
from sklearn.preprocessing import LabelEncoder


# Loading data
df= pd.read_csv('D:\\Yusuf\\dataset\\both_datasets.csv')

#==========================================# count numbers of instances per class

cnt = Counter(df.label)
print("label count = ", cnt)
# select only 2 most common classes!
top_classes = 2
# sort classes
sorted_classes = cnt.most_common()[:top_classes]
classes = [c[0] for c in sorted_classes]
counts = [c[1] for c in sorted_classes]
print("at least " + str(counts[-1]) + " instances per class")

#apply to dataframe
print(str(df.shape[0]) + " instances before")
df = df[[c in classes for c in df.label]]
print(str(df.shape[0]) + " instances after")

seqs = df.sequence.values
lengths = [len(s) for s in seqs]

# visualize
#fig, axarr = plt.subplots(1,2, figsize=(20,5))
#axarr[0].bar(range(len(classes)), counts)
#plt.sca(axarr[0])
#plt.xticks(range(len(classes)), classes, rotation='vertical')
#axarr[0].set_ylabel('frequency')
#
#axarr[1].hist(lengths, bins=100, normed=False)
#axarr[1].set_xlabel('sequence length')
#axarr[1].set_ylabel('# sequences')
#plt.show()

#=================================================
#Transform Labels
from keras.utils import to_categorical
Y = to_categorical(df.label)

#lb = LabelBinarizer()
#Y = lb.fit_transform(df.label)
#print(Y[:7])
# maximum length of sequence, everything afterwards is discarded!
max_length = 512

#create and fit tokenizer
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(seqs)
#represent input data as word rank number sequences
X = tokenizer.texts_to_sequences(seqs)
X = sequence.pad_sequences(X, maxlen=max_length)

#PDB1075 Dataset==================================
X1 = X[1:1069]
Y1 = Y[1:1069]
#PDB186 Dataset
X2 = X[1071:]
Y2 = Y[1071:]
#=================================================
print("Length of tokens: ",len(tokenizer.word_index)+1)

embedding_dim = 32

# create the model
model = Sequential()
model.add(Embedding(len(tokenizer.word_index)+1, embedding_dim, input_length=max_length))
model.add(Conv1D(filters=64, kernel_size=6, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=.2)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)


train_pred = model.predict(X_train)
test_pred1 = model.predict(X_test)
print("PDB1075 train-acc = " + str(accuracy_score(np.argmax(y_train, axis=1), np.argmax(train_pred, axis=1))))
print("PDB1075 test-acc = " + str(accuracy_score(np.argmax(y_test, axis=1), np.argmax(test_pred1, axis=1))))

#==========Predicting PDB186 Dataset=====
test_pred2 = model.predict(X2)
print("\n\n PDB 186 test-acc = " + str(accuracy_score(np.argmax(Y2, axis=1), np.argmax(test_pred2, axis=1))))
#========================================
# Compute confusion matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(test_pred, axis=1))

# Plot normalized confusion matrix
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
np.set_printoptions(precision=2)
plt.figure(figsize=(10,10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()

#===========
lb = LabelBinarizer()
Y = lb.fit_transform(df.label)

tick_marks = np.arange(len(lb.classes_))
plt.xticks(tick_marks, lb.classes_, rotation=90)
plt.yticks(tick_marks, lb.classes_)
#for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#    plt.text(j, i, format(cm[i, j], '.2f'), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2. else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#print(classification_report(np.argmax(y_test, axis=1), np.argmax(test_pred, axis=1), target_names=lb.classes_))

#Note=============================
#This result suggest the applicability of the NLP-theory to protein sequence classification. 
#Further improvements are possible (deeper model, smaller sequences, using n-grams as words for word embedding to accelerate learning)
# and should be validated via cross-validation. Another major concern is about overfitting small training data, but this could be 
# overcome by famous techniques like Dropout or regularizing by adding penalties for large weights (kernel_regularizer) or 
# large activations (activation_regularizer).





