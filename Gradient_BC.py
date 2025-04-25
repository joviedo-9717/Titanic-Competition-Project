#Titanic Kagle Project Gradient Boosting Classifier
#Programmer: Jennifer Oviedo

import pandas as pd
from sklearn.impute import SimpleImputer

# Load data
train = pd.read_csv('train.csv')
train.head()
test = pd.read_csv('test.csv')   
test.head()

# Exploratory Data Analysis
print(train.shape)  # Out: (891, 12) --> 891 rows x 12 columns
print(train.describe())
print(train.isnull().sum())  

print()
print()
passenger_num = train['Sex'].value_counts()
print(passenger_num)

# Sex
# 0    577
# 1    314

# Count survivors by gender
survivors = train[train['Survived'] == 1]['Sex'].value_counts()

# Calculate survival rates
male_survival_rate = survivors[0] / passenger_num[0]
female_survival_rate = survivors[1] / passenger_num[1]

print(f"Survival rate for males: {male_survival_rate:.2%}")
print(f"Survival rate for females: {female_survival_rate:.2%}")
# Name: count, dtype: int64
# Survival rate for males: 18.89%
# Survival rate for females: 74.20% 

# Data Pre-processing

# Categorical into numerical 
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male':0, 'female':1})    
train['Embarked'] = train['Embarked'].map({'S': 0, 'C':1, 'Q':2})
test['Embarked'] = test['Embarked'].map({'S': 0, 'C':1, 'Q':2})

# Impute - Replace NULL with the age average
age_imputer = SimpleImputer(strategy="mean")
train['Age'] = age_imputer.fit_transform(train[['Age']])
test['Age'] = age_imputer.transform(test[['Age']])                

fare_imputer = SimpleImputer(strategy="most_frequent")
train['Fare'] = fare_imputer.fit_transform(train[['Fare']])
test['Fare'] = fare_imputer.transform(test[['Fare']])

embarked_imputer = SimpleImputer(strategy="most_frequent")
train['Embarked'] = embarked_imputer.fit_transform(train[['Embarked']])
test['Embarked'] = embarked_imputer.transform(test[['Embarked']])

# New column in Data Frame - Family size
train['Family Size'] = train['SibSp'] + train['Parch'] + 1
test['Family Size'] = test['SibSp'] + test['Parch'] + 1


# Remove features - Training
train_predictors = train.drop(columns = ['Survived','Name','PassengerId','Ticket', 
                                         'Cabin', 'SibSp', 'Parch'])

# Target Variable
train_target = train['Survived']

# Remove features - Testing
test_predictors = test.drop(columns = ['PassengerId','Name','Ticket','Cabin',
                                       'SibSp', 'Parch'])

# Check Missing values
print(train_predictors)
print(train_predictors.isnull().sum())   
print(test_predictors.isnull().sum())

# Gradient Boosting 
from sklearn.ensemble import GradientBoostingClassifier

# Initialize model
gb_model = GradientBoostingClassifier(
            n_estimators = 100,
            learning_rate=0.1,
            max_depth=3,        
            random_state=42)

# Training the model
gb_model.fit(train_predictors, train_target)

# Predict survival using test.csv
gb_predictions = gb_model.predict(test_predictors)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(gb_model, train_predictors, train_target, cv=10)
print(f"Gradient Boosting Classifier: Mean Accuracy={scores.mean():.2%} | Std Dev = {scores.std():.2%}")


# Features used the most
import matplotlib.pyplot as plt

importances = gb_model.feature_importances_
features = train_predictors.columns

# Plot
plt.figure(figsize=(8,5))
plt.barh(features, importances)
plt.title('Feature Importance (Gradient Boosting)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_predictors, train_target, test_size=0.2, random_state=42)
gb_model.fit(X_train, y_train)
y_pred = gb_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Random Forest")
plt.show()


# Create a submission DataFrame
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': gb_predictions})


# Save to CSV
submission.to_csv("GradientBoosting_submission.csv", index=False)
print("Submission file created: GradientBoosting_submission.csv")

