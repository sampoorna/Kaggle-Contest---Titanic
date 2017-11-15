# Kaggle Contest - Titanic
Scripts for the Titanic Survivors Kaggle contest (https://www.kaggle.com/c/titanic) along with sample training and test data

Observations:
-------------

1. Using SVM
   - With rbf kernel
   - Gamma = 1 overfits more than 0.1 and 0.01
   - Gamma = 0.001 is better than 1
   - Incorporating grid search to find the best C and gamma
     - The best gamma is consistently 0.001
     - The best C is in the range of 10^3 but varies from 300-500
     - Chosen C = 560 based on best performance on Kaggle