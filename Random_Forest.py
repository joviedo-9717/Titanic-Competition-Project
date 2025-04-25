#Titanic Kagle Project Random Forest Regression
#Programmer: Jennifer Oviedo

import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load data
train = pd.read_csv('train.csv')
train.head()

test = pd.read_csv('test.csv')  
#test.head()

# Exploratory Data Analysis
# Check Missing values
print(train.shape)  # Out: (891, 12) --> 891 rows x 12 columns
print(train.describe())
print(train.isnull().sum())   
# Output:
    # [8 rows x 7 columns]
    # PassengerId      0
    # Survived         0
    # Pclass           0
    # Name             0
    # Sex              0
    # Age            177
    # SibSp            0
    # Parch            0
    # Ticket           0
    # Fare             0
    # Cabin          687
    # Embarked         2
    
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

fare_imputer = SimpleImputer(strategy="mean")
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


from sklearn.ensemble import RandomForestClassifier

f_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
f_model.fit(train_predictors, train_target)
forest_predictions = f_model.predict(test_predictors)


from sklearn.model_selection import cross_val_score

scores = cross_val_score(f_model, train_predictors, train_target, cv=10)
print(f"Random Forest: Mean Accuracy={scores.mean():.2%} | Std Dev = {scores.std():.2%}")

# Analyze what features have the most impact on the model
importances = f_model.feature_importances_
plt.figure(figsize=(8,5))
plt.barh(train_predictors.columns, importances)
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_predictors, train_target,
                                                    test_size=0.2, random_state=42)
f_model.fit(X_train, y_train)
y_pred = f_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                              display_labels=["Did Not Survive", "Survived"])
disp.plot()
plt.title("Confusion Matrix - Random Forest")
plt.show()


submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': forest_predictions})
submission.to_csv('Random_Forest.csv', index=False)


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=['Did Not Survive', 'Survived']))
