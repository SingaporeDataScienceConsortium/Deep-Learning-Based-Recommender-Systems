'''
This work is prepared for the SDSC course titled
'Introduction to the Application of Data Science in Recommender Systems'
Data reference: https://archive.ics.uci.edu/ml/datasets/adult
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from keras.layers import Input, Embedding, Dense, Flatten, Activation, concatenate
from keras.layers.advanced_activations import ReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model

# categorical information
categorical_cols = ['workclass', 'education', 'marital_status', 'occupation',
                    'relationship','race', 'gender', 'native_country']

# continuous information
continuous_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

train_data = pd.read_csv('training_data.csv') # train 32561
train_data['label'] = 1
train_data.loc[train_data['income_bracket']==' <=50K', 'label'] = 0

#uncomment the following if you want a more balanced training dataset
#train_data = pd.concat([train_data, train_data[train_data['label']==1], train_data[train_data['label']==1]]).sample(frac=1)

test_data = pd.read_csv('test_data.csv') # test 16281
test_data['label'] = 1
test_data.loc[test_data['income_bracket']==' <=50K.', 'label'] = 0

all_data = pd.concat([train_data, test_data]) # combine train and test 48842

y = all_data['label'].values # get all labels

for c in categorical_cols:
    label_encoder = LabelEncoder()
    all_data[c] = label_encoder.fit_transform(all_data[c]) # convert all categories to category numbers (0-n)

# split into training and test datasets
train_size = len(train_data)
x_train = all_data.iloc[:train_size]
y_train = y[:train_size]
x_test = all_data.iloc[train_size:]
y_test = y[train_size:]

# split categorical and continuous information for training datasets (8 columns)
x_train_categ = np.array(x_train[categorical_cols])
x_train_conti = np.array(x_train[continuous_cols])

# split categorical and continuous information for test datasets (5 columns)
x_test_categ = np.array(x_test[categorical_cols])
x_test_conti = np.array(x_test[continuous_cols])

# Standardize continuous features by removing the mean and scaling to unit variance
scaler = StandardScaler()
x_train_conti = scaler.fit_transform(x_train_conti)
x_test_conti = scaler.transform(x_test_conti)

# Generate polynomial and interaction features
poly = PolynomialFeatures(degree=2, interaction_only=True)
x_train_categ_poly = poly.fit_transform(x_train_categ)
x_test_categ_poly = poly.transform(x_test_categ)

# deep part
categorical_inputs = [] # construct a list of inputs for categorical information
categorical_embedding = [] # construct a list of embeddings for categorical information
for i in range(len(categorical_cols)): # for each categorical information
    input_i = Input(shape=(1,), dtype='int32')
    dim = len(np.unique(all_data[categorical_cols[i]])) # number of classes
    embed_dim = 5 # feature vector of length 5
    embed_i = Embedding(dim, embed_dim)(input_i) # embedding
    flatten_i = Flatten()(embed_i) # dimension follows conti_dense below

    categorical_inputs.append(input_i)
    categorical_embedding.append(flatten_i)

conti_input = Input(shape=(len(continuous_cols),))
conti_dense = Dense(256)(conti_input)

conti_and_embedding = concatenate([conti_dense]+categorical_embedding) # 256 + 5*8

# fully-connected layers
conti_and_embedding = Activation('relu')(conti_and_embedding)
conti_embedding_bn = BatchNormalization()(conti_and_embedding)
fc1 = Dense(512, use_bias=False)(conti_embedding_bn)
ac1 = ReLU()(fc1)
bn1 = BatchNormalization()(ac1)
fc2 = Dense(256, use_bias=False)(bn1)
ac2 = ReLU()(fc2)
bn2 = BatchNormalization()(ac2)
fc3 = Dense(128)(bn2)
ac3 = ReLU()(fc3)
deep_component = ac3

# wide part
dim = x_train_categ_poly.shape[1] # 37 features from 8 components
wide_component = Input(shape=(dim,))

# create
inputs = [conti_input] + categorical_inputs + [wide_component] # deep part + wide part
combine_layer = concatenate([deep_component, wide_component])
output = Dense(1, activation='sigmoid')(combine_layer) # output

model = Model(inputs=inputs, outputs=output)

# train
input_data = [x_train_conti] + [x_train_categ[:, i] for i in range(x_train_categ.shape[1])] + [x_train_categ_poly]
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(input_data, y_train, epochs=15, batch_size=128)

# evaluate
input_data = [x_test_conti] + [x_test_categ[:, i] for i in range(x_test_categ.shape[1])] + [x_test_categ_poly]
loss, accuracy = model.evaluate(input_data, y_test)
print('Accuracy:', np.round(100*accuracy, 2), '%')

# save
model.save('model/wide_and_deep')



