'''
This work is prepared for the SDSC course titled
'Introduction to the Application of Data Science in Recommender Systems'
Data reference: https://www.kaggle.com/zygmunt/goodbooks-10k
'''

import numpy as np
import pandas as pd
import random
from keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv('ratings.csv') # load data

model = load_model('model/Model_Embedding') # load model

predictions = model.predict([dataset.user_id, dataset.book_id])

matches = (dataset.rating - np.round(predictions).flatten()).value_counts()[0] # check matches
print('Accuracy for all data:', np.round(100*matches/len(predictions), 2), '%' )

# check matches with some tolerance
matches_with_tolerance = (dataset.rating - np.round(predictions).flatten()).value_counts()[0] + \
                         (dataset.rating - np.round(predictions).flatten()).value_counts()[1] + \
                         (dataset.rating - np.round(predictions).flatten()).value_counts()[-1]
print('Accuracy for all data (with tolerance):', np.round(100*matches_with_tolerance/len(predictions), 2), '%' )

inds = random.choices(list(dataset.index), k=10) # randomly choose some data

predictions = model.predict([dataset.user_id[inds], dataset.book_id[inds]])
print('\nPrediction VS Label')
_ = [print(np.round(predictions[i, 0], 2), 'VS', dataset.rating[inds[i]]) for i in range(len(inds))]

