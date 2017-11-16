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
print "Reading input files: COMPLETE"

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

### Extract titles, replace uncommon variations and add as column to dataframe
names_train = train_df['Name'].tolist()
names_test = test_df['Name'].tolist()
titles = [name.split(',')[1].split(' ')[1] for name in names_train]
train_df['Title'] = titles
titles = [name.split(',')[1].split(' ')[1] for name in names_test]
test_df['Title'] = titles
train_df.Title.replace(['Ms.', 'Mlle.', 'Mme.'], ['Miss.', 'Miss.', 'Mrs.'], inplace=True)
train_df.Title.replace(['Dona.', 'Capt.', 'Col.', 'Don.', 'Dr.', 'Jonkheer.', 'Lady.', 'Major.', 'Rev.', 'Sir.', 'the'], ['Other', 'Mr.', 'Mr.', 'Other', 'Other', 'Mr.', 'Other', 'Mr.', 'Mr.', 'Mr.', 'Other'], inplace=True) 
test_df.Title.replace(['Ms.', 'Mlle.', 'Mme.'], ['Miss.', 'Miss.', 'Mrs.'], inplace=True)
test_df.Title.replace(['Dona.', 'Capt.', 'Col.', 'Don.', 'Dr.', 'Jonkheer.', 'Lady.', 'Major.', 'Rev.', 'Sir.', 'the'], ['Other', 'Mr.', 'Mr.', 'Other', 'Other', 'Mr.', 'Other', 'Mr.', 'Mr.', 'Mr.', 'Other'], inplace=True)

### Replace non-numeric values with numbers
train_df.Sex.replace(['male', 'female'], [0, 1], inplace=True)
test_df.Sex.replace(['male', 'female'], [0, 1], inplace=True)

train_df.Title.replace(['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Other'], [0, 1, 2, 3, 4], inplace=True)
test_df.Title.replace(['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Other'], [0, 1, 2, 3, 4], inplace=True)

train_df.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2], inplace=True)
test_df.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2], inplace=True)
print "Initial data processing: COMPLETE"

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
print "Train/test split: COMPLETE"

#######################################################################################
### Parameter tuning using grid-search
C_range = np.logspace(1, 5, 20)
gamma_range = np.logspace(-3, 3, 13)
print C_range
param_grid = dict(C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(kernel = 'rbf', gamma=0.001), param_grid=param_grid, cv=cv)
grid.fit(X, y)
print "The best parameters are ", grid.best_params_, "with a score of", grid.best_score_

### Build model
clf = svm.SVC(kernel = 'rbf', gamma = 0.001, C = 560, cache_size=500) # 1 overfits more than 0.1 and 0.01
clf.fit(X, y)
print "Model learning: COMPLETE"
score = clf.score(X, y)
print score
predictions = clf.predict(X_pred)

write_file = "output_svm.csv"
with open(write_file, "wb") as output:
	writer = csv.writer(output, delimiter=',')
	writer.writerow(['PassengerId','Survived'])
	for ind in range(len(predictions)):
		writer.writerow([ids[ind], predictions[ind]])
print "Writing results to file: COMPLETE"
