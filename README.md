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
   - Normalize titles in names
     - Convert 'Mlle.' and 'Ms.' to 'Miss.', and 'Mme.' to 'Mrs.'
	 - Convert some rare titles ('Capt.', 'Col.', 'Rev.') to 'Mr.', which are observed to be only male in this dataset (not being sexist), and would most likely require one to be an adult
	 - Club others into a separate class 'Other'
   - Replace NaN values
     - For Age and Fare, replace by median
	 - For Embarked, only 2 rows in the training set have NaN here. We choose 'C'  (Charbourg as port of embarkation) as they paid $80 for first class travel that coincides with how much others from that port and that class paid. ---> This gets us the best score yet
   
	 