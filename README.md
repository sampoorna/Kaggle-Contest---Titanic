# Kaggle Contest - Titanic
Scripts for the Titanic Survivors Kaggle contest (https://www.kaggle.com/c/titanic) along with sample training and test data

Observations:
-------------

1. Using SVM
   - With rbf kernel
   - Gamma = 1 overfits more than 0.1 and 0.01
   - Gamma = 0.001 is better than 1
   - Incorporating grid search to find the best C and gamma finds best gamma = 0.001 and C = 562
   