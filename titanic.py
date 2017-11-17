import numpy as np
import csv
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pandas as pd
import random
import math

def applyMother(row):
	if row['Age'] >= 18 and row['Title'] != 'Miss' and row['Parch'] and row['Sex'] == 'female' > 0:
		return 1
	return 0
		
def applyMedianFare(row, df):
	#print row['Fare']
	if math.isnan(row['Fare']):
		return df[(df.Pclass == row['Pclass']) & (df.Embarked == row['Embarked'])]['Fare'].median()
	return row['Fare']
		
### Read files
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
print "Reading input files: COMPLETE"

### Save passenger IDs to write to file later
ids = test_df['PassengerId'].tolist()

### Training labels
y = train_df['Survived'].tolist()

combined = [train_df, test_df]

for dataset in combined:
	### Drop ticket info as it is not likely to be very informative
	dataset.drop(['Ticket'], axis = 1)

	### Replace NaN values
	dataset.fillna({'Age' : dataset['Age'].median(), 'Embarked' : 'C'}, inplace = True)
	dataset['Fare'] = dataset.apply(applyMedianFare, df=dataset, axis=1)

	### Extract titles, replace uncommon variations and add as column to dataframe
	dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
	dataset.Title.replace(['Ms', 'Mlle', 'Mme'], ['Miss', 'Miss', 'Mrs'], inplace=True)
	dataset.Title.replace(['Dona', 'Capt', 'Col', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir', 'Countess'], ['Other', 'Mr', 'Mr', 'Other', 'Other', 'Mr', 'Other', 'Mr', 'Mr', 'Mr', 'Other'], inplace=True) 

	### Create new feature Family_Size
	dataset['Family_Size'] = dataset['SibSp'] + dataset['Parch'] + 1

	### Create new feature Mother
	dataset['Mother'] = dataset.apply(applyMother, axis=1)

	### Replace non-numeric values with numbers
	dataset.Sex.replace(['male', 'female'], [0, 1], inplace=True)
	dataset.Title.replace(['Mr', 'Mrs', 'Miss', 'Master', 'Other'], [0, 1, 2, 3, 4], inplace=True)
	dataset.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2], inplace=True)
	print "Initial data processing: COMPLETE"

### Transform dataframe to lists
#X = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'Family_Size']].values.tolist()
#X_pred = test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'Family_Size']].values.tolist()
X = train_df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Parch', 'SibSp', 'Family_Size', 'Mother', 'Title']].values.tolist()
X_pred = test_df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Parch', 'SibSp', 'Family_Size', 'Mother', 'Title']].values.tolist()

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
#C_range = np.logspace(2, 5, 10)
#gamma_range = np.logspace(-3, 3, 13)
#print C_range
#param_grid = dict(C=C_range)
#cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
#grid = GridSearchCV(SVC(kernel = 'rbf', gamma=0.001), param_grid=param_grid, cv=cv)
#grid.fit(X, y)
#print "The best parameters are ", grid.best_params_, "with a score of", grid.best_score_

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
