'''
This work is prepared for the SDSC course titled
'Introduction to the Application of Data Science in Recommender Systems'
Data reference: https://archive.ics.uci.edu/ml/datasets/adult
'''

import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder


# data preparation starts
categorical_cols = ['workclass', 'education', 'marital_status', 'occupation',
                    'relationship','race', 'gender', 'native_country']

continuous_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

train_data = pd.read_csv('training_data.csv') # train 32561
train_data['label'] = 1
train_data.loc[train_data['income_bracket']==' <=50K', 'label'] = 0

test_data = pd.read_csv('test_data.csv') # test 16281
test_data['label'] = 1
test_data.loc[test_data['income_bracket']==' <=50K.', 'label'] = 0

all_data = pd.concat([train_data, test_data]) # combine train and test 48842


y = all_data['label'].values
for c in categorical_cols:
    label_encoder = LabelEncoder()
    all_data[c] = label_encoder.fit_transform(all_data[c]) # conver all categories to category numbers
train_size = len(train_data)
x_train = all_data.iloc[:train_size]
y_train = y[:train_size]
x_test = all_data.iloc[train_size:]
y_test = y[train_size:]

x_train_categ = np.array(x_train[categorical_cols])
x_test_categ = np.array(x_test[categorical_cols])
x_train_conti = np.array(x_train[continuous_cols])
x_test_conti = np.array(x_test[continuous_cols])

scaler = StandardScaler()
x_train_conti = scaler.fit_transform(x_train_conti)
x_test_conti = scaler.transform(x_test_conti)

poly = PolynomialFeatures(degree=2, interaction_only=True)
x_train_categ_poly = poly.fit_transform(x_train_categ)
x_test_categ_poly = poly.transform(x_test_categ)
# data preparation ends

# load model
model = load_model('model/wide_and_deep')

input_data = [x_test_conti] + [x_test_categ[:, i] for i in range(x_test_categ.shape[1])] + [x_test_categ_poly]
loss, accuracy = model.evaluate(input_data, y_test)
print('Accuracy:', np.round(100*accuracy, 2), '%')


















