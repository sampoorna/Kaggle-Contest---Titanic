# Kaggle Contest - Titanic
Scripts for the Titanic Survivors Kaggle contest (https://www.kaggle.com/c/titanic) along with sample training and test data

Observations:
-------------

1. Using SVM
   - With RBF kernel
   - Gamma = 1 overfits more than 0.1 and 0.01
   - Gamma = 0.001 is better than 1
   - Incorporating grid search to find the best C and gamma
     - The best gamma is consistently 0.001
     - The best C is in the range of 10^3 but varies from 300-500
     - Chosen C = 560 based on best performance on Kaggle
	 
   - Using polynomial kernel
     - Using C = 560 as with RBF kernel
     - Degree = 3 doesn't perform as well as RBF with params above
	 - Degree = 5 takes too long to run
	 
2. Feature engineering
   - Create new feature Title, extract and normalize the titles from Name
     - Convert 'Mlle' and 'Ms' to 'Miss', and 'Mme' to 'Mrs'
	 - Convert some rare titles ('Capt', 'Col', 'Rev') to 'Mr', which are observed to be only male in this dataset (not being sexist), and would most likely require one to be an adult
	 - Convert titles indicative of royalty/nobility ('Lady', 'Sir', 'Countess') to class 'Royal'
	 - Club others into a separate class 'Other'
   - Replace NaN values
     - For Age replace by median
	 - For Fare, replace by median of the same class and port of embarkation
	 - For Embarked, only 2 rows in the training set have NaN here. We choose 'C'  (Charbourg as port of embarkation) as they paid $80 for first class travel that coincides with how much others from that port and that class paid. ---> This gets us the best score yet
   - Create new feature Family_Size based on Sibsp and Parch
     - Family_Size = Sibsp + Parch + 1 (to include the passenger themselves)
	 - Adding Family_Size & Title does not yet improve upon the performance
	 - Adding only Title is better than adding Family_Size & Title
	 - Replacing SibSp & Parch with Family_Size & Title is worse
   - Create new feature Mother based on age, title and Parch
     - Mother = 1 if their title is not Miss., if Age >= 18, if Parch > 0 and if Sex is female
	 - Adding Mother, Family_Size & Title does not yet improve performance, but is better than only adding Family_Size & Title and same as only using Title
	 - Using ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Parch', 'SibSp', 'Mother'] is worse than not using Mother, but better than when Family_Size and Title replaced Sibsp and Parch
	 - Using ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Parch', 'SibSp', 'Mother', 'Title'] gives us the same error as without Mother ---> Negligible added utility of including Mother
   - Create new feature CabinBool
     - Boolean value, = 1 if Cabin has a recorded value, 0 otherwise
	 - Gives lower score than using ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Parch', 'SibSp', 'Mother', 'Title']