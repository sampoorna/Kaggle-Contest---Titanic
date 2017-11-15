import numpy as np
import csv
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
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

### Parameter tuning using grid-search
C_range = np.logspace(0, 5, 13)
gamma_range = np.logspace(-6, 3, 13)
print C_range
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, y)

print "The best parameters are ", grid.best_params_, "with a score of", grid.best_score_

### Build model
clf = svm.SVC(kernel = 'rbf', gamma = grid.best_params_['gamma'], C = grid.best_params_['C'], cache_size=500) # 1 overfits more than 0.1 and 0.01
clf.fit(X, y)
score = clf.score(X, y)
print score
predictions = clf.predict(X_pred)

print predictions

write_file = "output_svm.csv"
with open(write_file, "wb") as output:
	writer = csv.writer(output, delimiter=',')
	writer.writerow(['PassengerId','Survived'])
	for ind in range(len(predictions)):
		writer.writerow([ids[ind], predictions[ind]])

