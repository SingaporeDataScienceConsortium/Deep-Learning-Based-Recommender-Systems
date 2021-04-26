'''
This work is prepared for the SDSC course titled
'Introduction to the Application of Data Science in Recommender Systems'
Data reference: https://grouplens.org/datasets/movielens/
'''

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# read data and split into training and test sets
dataset = pd.read_csv('ratings_100k.csv')
dataset = dataset.drop(columns = ['Unnamed: 0']) # remove unwanted columns
train, test = train_test_split(dataset, test_size=0.2)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

n_users = len(set(dataset['user'].tolist())) # total number of users
n_movies = len(set(dataset['movie'].tolist())) # total number of movies
n_features = 50

# initialize feature vectors for users and movies
P = np.random.rand(n_users, n_features) # (963, 50)
Q = np.random.rand(n_movies, n_features) # (1682, 50)


#  ======== training parameters ========
epochs = 5 # number of iterations
theta = 0.0001 # threshold for stop training
alpha = 0.02 # learning rate
lambda_  = 0.02 # regularization term

previous_error = 0.0 # initialize previous loss
cost_of_epoch=[] # all cost
for epoch in range(epochs):
    print('epoch', epoch)
    current_error = 0.0 # initialize current loss

    for i in range(train.shape[0]): # read through all records
        temp = train.iloc[i]
        u, m, r = int(temp['user']), int(temp['movie']), temp['rating']
        pred = np.dot(P[u],Q[m]) # predicted rating
        err = r-pred # error for a single record
        P[u] = P[u] + alpha * (err * Q[m] - lambda_ * P[u]) # update user features
        Q[m] = Q[m] + alpha * (err * P[u] - lambda_ * Q[m]) # update movie features

        # error term and regularization term
        current_error = current_error + pow(err,2) + lambda_ * (sum(pow(P[u], 2)) + sum(pow(Q[m], 2)))

    cost_of_epoch.append(current_error) # records total loss for each epoch
    print('cost', np.round(current_error, 2))
    if abs(current_error - previous_error) < theta: # convergence; loss does not change much
        break # stop training if loss stablize
    previous_error = current_error # update previous loss
    alpha = alpha * 0.9 # decay learning rate


# ======== show losses ========
n = range(len(cost_of_epoch))
plt.plot(n, cost_of_epoch)
plt.title('Costs')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()


# ======== save features of users and movies ========
data = {'P':P, 'Q':Q}
f = open('FunkSVD.pkl', 'wb')
pickle.dump(data, f)


# ======== read saved features of users and movies ========
f = open('FunkSVD.pkl', 'rb')
features = pickle.load(f)
P = features['P']
Q = features['Q']


# ======== test dataset ========
rmse = 0 # initialize rmse
n = test.shape[0]
for i in range(test.shape[0]):
    temp = train.iloc[i]
    u, m, r = int(temp['user']), int(temp['movie']), temp['rating']
    pred = np.dot(P[u],Q[m]) # predicted ratings
    rmse = rmse + pow((r-pred),2) # accumulate errors
rmse = (rmse/n)**0.5
print("rmse:", np.round(rmse, 3))


# ======== recommendation ========
user = 666
k = 10 # recommend 10 movies
user_id = user-1
pred = np.dot(P[user_id], Q.transpose()) #predicted ratings for all movies including those watched by the user
movies_recommend = np.argsort(pred)[::-1] # sorted movie indices

# sorted predicted movie ratings
pred.sort()
pred = pred[::-1]

print('\nmovie_index    rating')
for i in range(k):
    print('    ' + str(movies_recommend[i]) + '        ' + str(np.round(pred[i], 2)))



