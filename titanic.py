import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
training_data = pd.read_csv("/home/lt88/python-projects/datasets/titanic/train.csv")
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

# Training data
labels_train = training_data.pop('Survived')
features_train = training_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].fillna(training_data.mean())
# print(features_train['Sex'].isna().values.any())
features_train = pd.get_dummies(features_train, drop_first=True)
print(features_train)

# Label encoding categorical feature Sex of train data
# label_encoder = LabelEncoder()
# label_encoder.fit_transform(features_train['Sex'])
# features_train['Sex'] = label_encoder.transform(features_train['Sex'])

# print(features_train)

# Testing data
testing_data = pd.read_csv("/home/lt88/python-projects/datasets/titanic/test.csv")
features_test = testing_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].fillna(testing_data.mean())
features_test = pd.get_dummies(features_test, drop_first=True)
print(features_test)

# Label encoding categorical feature Sex of test data
# label_encoder.fit(features_test['Sex'])
# features_test['Sex'] = label_encoder.transform(features_test['Sex'])

clf = RandomForestClassifier()
clf.fit(features_train, labels_train)

prediction = clf.predict(features_test)

prediction_dataframe = pd.DataFrame(prediction)

passenger_ids = testing_data.loc[:, "PassengerId"]

prediction_dataframe.insert(0, 'PassengerId', passenger_ids)
prediction_dataframe.columns = ['PassengerId', 'Survived']

prediction_dataframe.to_csv("/home/lt88/python-projects/datasets/titanic/prediction.csv", index=False)

print(prediction_dataframe)
