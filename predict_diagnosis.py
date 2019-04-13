import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

data = pd.read_csv('Training.csv')
df_symptoms_on_russian = pd.read_csv('simptom_columns_question_4.0.csv', sep=',', header=None,
                                         names=['symptom', 'question'])
df_symptoms_on_russian['symptom'] = df_symptoms_on_russian['symptom'].str.strip()

X_train_data = data.drop(['prognosis'], axis= 1)
y_train_data = data['prognosis']

def filter_df(X, y, categories):
    for categoria, value in categories.items():
        if categoria in X.columns.values:
            y = y[X[categoria] == value]
            X = X[X[categoria] == value]
        else:
            return [], []
    return X, y

def predict(categories):
    data = {}
    y_train = y_train_data
    X_train = X_train_data

    X_train, y_train = filter_df(X_train, y_train, categories)

    if len(np.unique(y_train)) == 0 or len(y_train) == 0:
        return "{'error': 'Not diagnosis in data base'}"

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(y_train)
    y_train_le = label_encoder.transform(y_train)

    if len(np.unique(y_train)) == 1:
        founded_diagnosis = label_encoder.classes_[0]
        data['diagnosis'] = founded_diagnosis

        return data

    tree = DecisionTreeClassifier(random_state=1).fit(X_train, y_train_le)

    index_feature_in_tree = tree.tree_.feature[0]
    x_columns = X_train.columns
    symptom_to_ask_user = x_columns[index_feature_in_tree]

    question_to_user = str(df_symptoms_on_russian[df_symptoms_on_russian['symptom'] == symptom_to_ask_user]['question'].values[0])
    data['question'] = question_to_user
    data['question_ref'] = symptom_to_ask_user
    return data

if __name__=="__main__":
    print(predict({'excessive_hunger': 1}))
