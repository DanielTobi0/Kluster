import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import joblib
from category_encoders import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('Crop_Data.csv')

# Features and Target
X_ = df.drop(['label'], axis=1)
y = df['label']

# OHE
encoder = OneHotEncoder(cols=['season', 'Country'])
X = encoder.fit_transform(X_)
joblib.dump(encoder, 'encoder.joblib')

# Column rename
X.rename(columns={'water availability' : 'water_availability'}, inplace=True)

# Label Encoder on target
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
le_mapping_names = dict(zip(le.classes_, le.transform(le.classes_)))

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=df['label'], random_state=1)

# LightGBM
base_model = lgb.LGBMClassifier(verbose=-1)

base_model.fit(X_train, y_train)
y_pred = base_model.predict(X_test)
metrics.accuracy_score(y_test, y_pred)

# Save model
FILENAME = 'base_model.pkl'
pickle.dump(base_model, open(FILENAME, 'wb'))

# Load model
pickled_model = pickle.load(open(FILENAME, 'rb'))
pickled_model.predict(X_test)
metrics.accuracy_score(y_test, y_pred)


# Train on outside data
row_to_predict = df.iloc[13:14, :]
row_to_predict.drop(['label'], axis=1, inplace=True)


# Make predictions on the selected row
load_encoder = joblib.load('encoder.joblib')
row_to_predict_ = load_encoder.transform(row_to_predict)

j = base_model.predict(row_to_predict_)
print(' '.join(j))


