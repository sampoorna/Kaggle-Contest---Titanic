
#######################################################################################
### Importing libraries
import numpy as np
import csv
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score
import pandas as pd
import random
import math

#######################################################################################
### Function definitions
def applyMother(row):
	if row['Age'] >= 18 and row['Title'] != 'Miss' and row['Parch'] and row['Sex'] == 'female' > 0:
		return 1
	return 0
		
def applyMedianFare(row, df):
	#print row['Fare']
	if math.isnan(row['Fare']):
		return df[(df.Pclass == row['Pclass']) & (df.Embarked == row['Embarked'])]['Fare'].median()
	return row['Fare']
	
def fillAge(row, df):
	if math.isnan(row['Age']):
		return df[(df.Title == row['Title'])]['Age'].mean()
	return row['Age']
	
def runKFold(clf, X_all, y_all):
	kf = KFold(n_splits=10)
	outcomes = []
	fold = 0
	for train_index, test_index in kf.split(X_all):
		#print train_index, test_index
		fold += 1
		X_train, X_test = X_all[train_index], X_all[test_index]
		y_train, y_test = y_all[train_index], y_all[test_index]
		clf.fit(X_train, y_train)
		predictions = clf.predict(X_test)
		accuracy = accuracy_score(y_test, predictions)
		outcomes.append(accuracy)
		#print("Fold {0} accuracy: {1}".format(fold, accuracy))     
	mean_outcome = np.mean(outcomes)
	#print("Mean Accuracy: {0}".format(mean_outcome))
	return mean_outcome

#######################################################################################	
### Read files
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
print "Reading input files: COMPLETE"

### Save passenger IDs to write to file later
ids = test_df['PassengerId'].tolist()

### Training labels
y = train_df['Survived'].tolist()

combined = [train_df, test_df]
#print pd.crosstab(train_df['CabinBool'], train_df['Survived'])

for dataset in combined:
	### Drop ticket info as it is not likely to be very informative
	dataset.drop(['Ticket'], axis = 1)

	### Replace NaN values
	#dataset.fillna({'Age' : dataset['Age'].median(), 'Embarked' : 'C'}, inplace = True)
	dataset.fillna({'Embarked' : 'C'}, inplace = True)
	dataset['Fare'] = dataset.apply(applyMedianFare, df=dataset, axis=1)
	
	### Binarize Cabin values to create new feature CabinBool
	dataset['CabinBool'] = dataset['Cabin'].notnull().astype('int')

	### Extract titles, replace uncommon variations and add as column to dataframe
	dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
	dataset.Title.replace(['Ms', 'Mlle', 'Mme'], ['Miss', 'Miss', 'Mrs'], inplace=True)
	dataset.Title.replace(['Dona', 'Capt', 'Col', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir', 'Countess'], ['Other', 'Mr', 'Mr', 'Other', 'Other', 'Mr', 'Royal', 'Mr', 'Mr', 'Royal', 'Royal'], inplace=True)
	
	### Replace Age NaN values
	dataset['Age'] = dataset.apply(fillAge, df=dataset, axis=1)
	
	### Create new feature AgeGroup
	bins = (0, 5, 12, 18, 25, 35, 60, 120)
	group_names = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
	group_values = [0, 1, 2, 3, 4, 5, 6]
	dataset['AgeGroup'] = pd.cut(dataset.Age, bins, labels=group_values)
	#print pd.crosstab(dataset['AgeGroup'], dataset['Title'])

	### Create new feature Family_Size
	dataset['Family_Size'] = dataset['SibSp'] + dataset['Parch'] + 1

	### Create new feature Mother
	dataset['Mother'] = dataset.apply(applyMother, axis=1)
	
	### Create new feature IsAlone
	dataset['IsAlone'] = dataset.Family_Size.map({1: 0})
	dataset.fillna({'IsAlone': 1}, inplace=True)

	### Replace non-numeric values with numbers
	dataset.Sex.replace(['male', 'female'], [0, 1], inplace=True)
	dataset.Title.replace(['Mr', 'Mrs', 'Miss', 'Master', 'Royal', 'Other'], [0, 1, 2, 3, 4, 5], inplace=True)
	dataset.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2], inplace=True)
print "Initial data processing: COMPLETE"

### Transform dataframe to lists
#X = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'Family_Size']].values.tolist()
#X_pred = test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'Family_Size']].values.tolist()
X = train_df[['Pclass', 'Sex', 'AgeGroup', 'SibSp', 'Parch', 'Fare', 'Embarked', 'IsAlone', 'Mother', 'Title', 'Family_Size']].values.tolist()
X_pred = test_df[['Pclass', 'Sex', 'AgeGroup', 'SibSp', 'Parch', 'Fare', 'Embarked', 'IsAlone', 'Mother', 'Title', 'Family_Size']].values.tolist()

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
#svm_param_grid = dict(C=C_range)
#svm_grid = GridSearchCV(SVC(kernel = 'rbf', gamma=0.001), param_grid=svm_param_grid, cv=cv)
#print "The best parameters are ", grid.best_params_, "with a score of", grid.best_score_

ada_param_grid = {"learning_rate" : [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 1.5],
			  "base_estimator__splitter" : ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" : [1, 2],
			  "base_estimator__criterion" : ["gini", "entropy"]}
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
ada_grid = GridSearchCV(AdaBoostClassifier(DecisionTreeClassifier(random_state=2)), param_grid=ada_param_grid, cv=cv)
ada_grid.fit(X, y)
print "The best parameters are ", ada_grid.best_params_, "with a score of", ada_grid.best_score_

#######################################################################################
### Building models
#clf = SVC(kernel = 'rbf', gamma = 0.001, C = 560, cache_size=500) # 1 overfits more than 0.1 and 0.01
random_state = 2
classifiers = []
classifiers.append({'Name': 'SVM', 'Model': SVC(kernel='rbf', gamma=0.001, C=560, cache_size=500, random_state=random_state)})
classifiers.append({'Name': 'Decision Tree', 'Model': DecisionTreeClassifier(random_state=random_state)})
classifiers.append({'Name': 'AdaBoost with Decision Tree', 'Model': AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=ada_grid.best_params_['learning_rate'], n_estimators=ada_grid.best_params_['n_estimators'], algorithm=ada_grid.best_params_['algorithm'], base_estimator__splitter=ada_grid.best_params_['base_estimator__splitter'], base_estimator__criterion=ada_grid.best_params_['base_estimator__criterion'])})
classifiers.append({'Name': 'Random Forest', 'Model': RandomForestClassifier(random_state=random_state)})
classifiers.append({'Name': 'Extra Trees', 'Model': ExtraTreesClassifier(random_state=random_state)})
classifiers.append({'Name': 'Gradient Boosting', 'Model': GradientBoostingClassifier(random_state=random_state)})
classifiers.append({'Name': 'MLP', 'Model': MLPClassifier(random_state=random_state)})
classifiers.append({'Name': 'KNN', 'Model': KNeighborsClassifier()})
classifiers.append({'Name': 'Logistic Regression', 'Model': LogisticRegression(random_state = random_state)})
#classifiers.append({'Name': 'LDA', 'Model': LinearDiscriminantAnalysis()})

### Run k-fold cross validation
for model in classifiers:
	print model['Name'], runKFold(model['Model'], np.array(X), np.array(y))

### Fit model on entire data for final predictions
#clf.fit(X, y)
#print "Model learning: COMPLETE"
# score = clf.score(X, y)
# print "Training error: ", score
# predictions = clf.predict(X_pred)

#######################################################################################
### Writing output to file
# write_file = "output_svm.csv"
# with open(write_file, "wb") as output:
	# writer = csv.writer(output, delimiter=',')
	# writer.writerow(['PassengerId','Survived'])
	# for ind in range(len(predictions)):
		# writer.writerow([ids[ind], predictions[ind]])
# print "Writing results to file: COMPLETE"
