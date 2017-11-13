import numpy as np
import csv
from sklearn import svm
import pandas as pd
import random

### Read files
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

### Save passenger IDs to write to file later
ids = test_df['PassengerId'].tolist()
#train_df.drop(['PassengerId', 'Ticket', 'Cabin', 'Name'], axis = 1)
#test_df.drop(['PassengerId', 'Ticket', 'Cabin', 'Name'], axis = 1)

### Replace NaN values
train_df.fillna({'Age' : train_df['Age'].median(), 'Embarked' : 'S', 
	'Fare' : train_df['Fare'].median()}, inplace = True)

test_df.fillna({'Age' : test_df['Age'].median(), 'Embarked' : 'S', 
    'Fare' : test_df['Fare'].median()}, inplace = True)

### Training labels
y = train_df['Survived'].tolist()
#train_df.drop(['Survived'], axis = 1)

### Replace non-numeric values with numbers
train_df.loc[train_df['Sex'] == 'male', 'Sex'] = 0
train_df.loc[train_df['Sex'] == 'female', 'Sex'] = 1
test_df.loc[test_df['Sex'] == 'male', 'Sex'] = 0
test_df.loc[test_df['Sex'] == 'female', 'Sex'] = 1

train_df.loc[train_df['Embarked'] == 'C', 'Embarked'] = 0
train_df.loc[train_df['Embarked'] == 'S', 'Embarked'] = 1
train_df.loc[train_df['Embarked'] == 'Q', 'Embarked'] = 2
test_df.loc[test_df['Embarked'] == 'C', 'Embarked'] = 0
test_df.loc[test_df['Embarked'] == 'S', 'Embarked'] = 1
test_df.loc[test_df['Embarked'] == 'Q', 'Embarked'] = 2

### Transform dataframe to lists
X = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values.tolist()
X_pred = test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values.tolist()

### Randomly shuffle training instances
combo = list(zip(X, y))
random.shuffle(combo)
X, y = zip(*combo)

### Split part of training set for validation
X_valid = X[0:int(0.1*len(X))]
y_valid = y[0:len(X_valid)]
X = X[(len(X_valid)+1):-1]
y = y[(len(X_valid)+1):-1]

### Build model
clf = svm.SVC(kernel = 'rbf', gamma = 1) # 1 works better than 0.1 and 0.01
clf.fit(X, y)
score = clf.score(X, y)
print score
predictions = clf.predict(X_pred)

print predictions

write_file = "Output.csv"
with open(write_file, "wb") as output:
	writer = csv.writer(output, delimiter=',')
	writer.writerow(["PassengerId,Survived"])
	for ind in range(len(predictions)):
		writer.writerow([ids[ind], predictions[ind]])

