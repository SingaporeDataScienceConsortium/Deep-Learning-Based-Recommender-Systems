'''
This work is prepared for the SDSC course titled
'Introduction to the Application of Data Science in Recommender Systems'
Data reference: https://www.kaggle.com/zygmunt/goodbooks-10k
'''

import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras.models import Model
import warnings
warnings.filterwarnings('ignore')

# ## Loading in data
dataset = pd.read_csv('ratings.csv') # read data

train, test = train_test_split(dataset, test_size=0.2) # split into training and test datasets

n_users = len(dataset.user_id.unique()) # number of unique customers

n_books = len(dataset.book_id.unique()) # number of unique customers


# create book embeddings
book_input = Input(shape=[1]) # input
book_embedding = Embedding(n_books+1, 32)(book_input) # embedding: 32 features
book_vector = Flatten()(book_embedding)

# create user embeddings
user_input = Input(shape=[1]) # input
user_embedding = Embedding(n_users+1, 32)(user_input) # embedding: 32 features
user_vector = Flatten()(user_embedding)


# concatenate features
concatenate = Concatenate()([book_vector, user_vector]) # 32 + 32 = 64 features
# print(model.summary()) # display the model architecture

# add fully-connected-layers
fc1 = Dense(32, activation='relu')(concatenate) # from 64 to 32
fc2 = Dense(16, activation='relu')(fc1) # from 32 to 16
out = Dense(1)(fc2) # from 16 to 1

# Create model and compile
model = Model([user_input, book_input], out)
model.compile(optimizer='adam', loss='mean_squared_error') # loss: mean squared error


model.fit([train.user_id, train.book_id], train.rating, batch_size=4096, epochs=5, verbose=1)
model.save('model/Model_Embedding') # save the model

predictions = model.predict([test.user_id, test.book_id]) # predict for test data

matches = (test.rating - np.round(predictions).flatten()).value_counts()[0] # check matches
print('Accuracy for test data:', np.round(100*matches/len(predictions), 2), '%' )

# check matches with some tolerance
matches_with_tolerance = (test.rating - np.round(predictions).flatten()).value_counts()[0] + \
                         (test.rating - np.round(predictions).flatten()).value_counts()[1] + \
                         (test.rating - np.round(predictions).flatten()).value_counts()[-1]
print('Accuracy for test data (with tolerance):', np.round(100*matches_with_tolerance/len(predictions), 2), '%' )

inds = random.choices(list(test.index), k=10) # randomly choose some test data

predictions = model.predict([test.user_id[inds], test.book_id[inds]])
print('\nPrediction VS Label (Test Data)')
_ = [print(np.round(predictions[i, 0], 2), 'VS', test.rating[inds[i]]) for i in range(len(inds))]






