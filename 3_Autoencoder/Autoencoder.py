'''
This work is prepared for the SDSC course titled
'Introduction to the Application of Data Science in Recommender Systems'
Data reference: https://grouplens.org/datasets/movielens/
'''

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
import random
import matplotlib.pyplot as plt
import keras.backend as K
import warnings
warnings.filterwarnings('ignore')

path = '' # data directory

movies_df = pd.read_csv(path + 'movies.csv', usecols=['movieId','title']) # read data with selective columns
movies_df = movies_df.set_index('movieId') # use movie ID as index
ratings_df=pd.read_csv(path + 'ratings.csv', usecols=['userId', 'movieId', 'rating']) # read data with selective columns

# create customer-movie table
user_movie_matrix_df = ratings_df.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)

user_movie_matrix = user_movie_matrix_df.values # convert to numerical matrix

num_input = user_movie_matrix.shape[1] # size of input
num_hidden_1 = 512 # size of hidden layer 1
num_hidden_2 = 128 # size of hidden layer 2

# network architecture: 9724-512-128-512-9724
input_data = Input(shape=(num_input,))
encoder1 = Dense(num_hidden_1, activation='sigmoid')(input_data)
encoder2 = Dense(num_hidden_2, activation='sigmoid')(encoder1)
decoder1 = Dense(num_hidden_1, activation='sigmoid')(encoder2)
output = Dense(num_input, activation=None)(decoder1)

@tf.function # compiled into tensorflow graph
def custom_loss1(y_true, y_pred):
    where = K.not_equal(y_true, 0.0) # get movies not rated yet
    y_pred = tf.multiply(y_pred, K.cast(where, tf.float32)) # mask non-rated movies as 0s
    loss = K.sum(K.square(y_pred - y_true)) # only check the difference for rated movies
    return loss


@tf.function # compiled into tensorflow graph
def custom_loss2(y_true, y_pred):
    loss = K.sum(K.square(y_pred - y_true)) # check the difference for the whole user-movie matrix
    return loss

autoencoder = Model(inputs=input_data, outputs=output)
autoencoder.compile(optimizer='rmsprop', loss=custom_loss1)

history = autoencoder.fit(user_movie_matrix, user_movie_matrix, epochs=100, batch_size=128, verbose=1) # fit

predictions = autoencoder.predict(user_movie_matrix) # predict

predictions_watched = np.round(predictions * 2) / 2 # round to 0.5
predictions_watched[user_movie_matrix==0] = 0 # mask movies not rated as 0s

pred = predictions_watched.flatten()
label = user_movie_matrix.flatten()

# get predictions and labels for all rated movies
pred = np.array([pred[i] for i in range(len(pred)) if label[i]!=0])
label = np.array([i for i in label if i!=0])

# difference
diff = label - pred

print('Prediction for watched movies:', np.round(100*len(diff[(diff==0)])/len(diff), 2), '%')
print('Prediction for watched movies (with tolerance):', np.round(100*len(diff[(np.abs(diff)<=1)])/len(diff), 2), '%')

#UserID = 1 # specify a customer
UserID = random.choice(list(user_movie_matrix_df.index)) # randomly choose a customer

# construct a dictionary: row index -> movie ID
ind2movieID_dict = {}
cols = list(user_movie_matrix_df.columns)
for i in range(len(cols)):
    ind2movieID_dict[i] = cols[i]


n_recommend = 10 # number of recommendations
user_watched = user_movie_matrix[UserID-1, :] # movie watched -> non-zero
user_pred = predictions[UserID-1, :] # rating prediction for this customer
user_pred[user_watched!=0] = 0 # label movies watched as 0s

sorted_movies = np.argsort(user_pred)[::-1] # sort movies not watched from high rating to low

print('\nRecommended Movies for User', UserID)
for i in range(n_recommend):
    print(i+1, movies_df.loc[ind2movieID_dict[sorted_movies[i]]]['title'])

# plot loss
plt.plot(history.history['loss'][3:])
plt.xlabel("Epochs")
plt.ylabel("Training Error")






