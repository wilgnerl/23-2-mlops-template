# Data
import pandas as pd

# Export
import pickle

# Plot
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

# LEIA O DATABASE
df = pd.read_csv()

# Defina X e Y
X = df
y = df
# COLOQUE AS COLUNAS CATEGORICAS
cat_cols = []

one_hot_enc = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore", drop="first"),
     cat_cols),
    remainder="passthrough")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1912)

X_train = one_hot_enc.fit_transform(X_train)
X_train = pd.DataFrame(X_train, columns=one_hot_enc.get_feature_names_out())

X_train.head(2)

X_test = pd.DataFrame(one_hot_enc.transform(
    X_test), columns=one_hot_enc.get_feature_names_out())
X_test.head(2)
model = LGBMClassifier()
model.fit(X_train, y_train)


# Specify the file path where you want to save the pickle file
file_path = "models/model.pkl"

# Save the model as a pickle file
with open(file_path, "wb") as f:
    pickle.dump(model, f)

file_path = "models/ohe.pkl"

# Save the OneHotEncoder as a pickle file
with open(file_path, "wb") as f:
    pickle.dump(one_hot_enc, f)
