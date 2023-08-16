import pickle
import pandas as pd

df = pd.read_csv("data/your_data_path")
df = df.drop(labels=["target_column"], axis=1)

# open a file, where you stored the pickled data
model_file = open('models/model_filename.pkl', 'rb')
ohe_file = open('models/ohe_filename.pkl', 'rb')

# dump information to that file
model = pickle.load(model_file)
ohe = pickle.load(ohe_file)

X = pd.DataFrame(ohe.transform(
    df), columns=ohe.get_feature_names_out())

y_pred = model.predict(X)


def convert_to_yes_no(value):
    return 'yes' if value == 1 else 'no'


y_pred_label = [convert_to_yes_no(element) for element in y_pred]

df["y_pred"] = y_pred_label

print(df["y_pred"].value_counts())

df.to_csv("data/data_predict_filename.csv")
