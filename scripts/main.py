import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

music_data = pd.read_csv('./data/music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X, y)

tree.export_graphviz(
  model, 
  out_file='music_recommender.dot', 
  feature_names=['age', 'gender'], 
  class_names=sorted(y.unique()), 
  label='all', 
  rounded=True, 
  filled=True
)
# model = joblib.load('music-recommender.joblib')
# prediction = model.predict([[ 21, 1 ]])
# print(prediction)


# score = accuracy_score(y_test, prediction)
# print(X_test)

# print(prediction)
# print(score)
