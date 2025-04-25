#Titanic Kagle Project
#Programmers: Jennifer Oviedo and Nathaniel Gosdin

import pandas as pd
from sklearn.impute import SimpleImputer
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train = pd.read_csv('train.csv')
train.head()

test = pd.read_csv('test.csv')     
#test.head()
# Exploratory Data Analysis

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

# Visualize survival by Gender
sns.countplot(x='Sex', hue='Survived', data=train)
plt.title('Survival by Gender', fontsize=12)
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Survived', labels=['No', 'Yes'])

# Visualize survival by Passenger Class
plt.figure(figsize=(8, 5)) 
sns.countplot(x='Survived', hue='Pclass', data=train)
plt.title('Survival by Passenger Class')
plt.legend(title='Survived', labels=['First', 'Second', 'Third'])
plt.show()

# Visualize survival by Age
ages = [0, 12, 18, 35, 60, 100]
age_labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']

age_group = pd.cut(train['Age'].dropna(), bins=ages, labels=age_labels)

age_group_df = pd.DataFrame({
    'Age_Group': age_group,
    'Survived': train.loc[train['Age'].notna(), 'Survived']
})


train = train.drop(columns=['Age_Group'], errors='ignore')


sns.countplot(x='Age_Group', hue='Survived', data=age_group_df)
plt.title('Survival by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()


# Data Pre-processing

# Categorical into numerical 
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male':0, 'female':1})    

# Impute - Replace NULL with the age average
age_imputer = SimpleImputer(strategy="mean")
train['Age'] = age_imputer.fit_transform(train[['Age']])
test['Age'] = age_imputer.transform(test[['Age']])                 

fare_imputer = SimpleImputer(strategy="mean")
train['Fare'] = fare_imputer.fit_transform(train[['Fare']])
test['Fare'] = fare_imputer.transform(test[['Fare']])

embarked_imputer = SimpleImputer(strategy="most_frequent")
train['Embarked'] = embarked_imputer.fit_transform(train[['Embarked']]).ravel()
test['Embarked'] = embarked_imputer.transform(test[['Embarked']]).ravel()


# Remove features - Training
train_predictors = train.drop(columns = ['PassengerId','Survived','Name','Ticket', 
                                         'Cabin', 'Embarked'])
# Target Variable
train_target = train['Survived']

# Remove features - Testing
test_predictors = test.drop(columns = ['PassengerId','Name','Ticket', 
                                         'Cabin', 'Embarked'])

# Check missing values 
print(train_predictors.isnull().sum())
print()
print(test_predictors.isnull().sum())   


# Selecting the best model based on accuracy score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Non scaled values - estimators
unscaled_estimators = {
    'Gaussian Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
}

# Run each model, print results, get CSV file
kfold = KFold(n_splits=10, random_state = 42, shuffle = True)
for name, model in unscaled_estimators.items():
    scores = cross_val_score(model, X=train_predictors, y=train_target, cv=kfold)
    print(f'{name:>25}: Mean Accuracy = {scores.mean():.2%} | Std Dev = {scores.std():.2%}')
    model.fit(train_predictors, train_target)
    predictions = model.predict(test_predictors)
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions})
    submission.to_csv(f"{name}_submission.csv", index=False)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_predictors = train_predictors.copy()
scaled_predictors[['Age', 'Fare']] = scaler.fit_transform(scaled_predictors[['Age', 'Fare']])
test_predictors[['Age', 'Fare']] = scaler.transform(test_predictors[['Age', 'Fare']])

# Scales values - estimators
scaled_estimators = {
    'Logistic Regression': LogisticRegression(),
    'Support Vector Classifier': SVC(gamma='scale'),
}

# Run each model, print results, get CSV file
for name, model in scaled_estimators.items():
    scores = cross_val_score(model, X=scaled_predictors, y=train_target, cv=kfold)  
    print(f'{name:>25}: Mean Accuracy = {scores.mean():.2%} | Std Dev = {scores.std():.2%}')
    model.fit(scaled_predictors, train_target)
    predictions = model.predict(test_predictors)    
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions})
    submission.to_csv(f"{name}_submission.csv", index=False)

# Output:
#      Gaussian Naive Bayes: Mean Accuracy = 79.01% | Std Dev = 4.70%
#             Decision Tree: Mean Accuracy = 77.21% | Std Dev = 2.86%
#             Random Forest: Mean Accuracy = 81.82% | Std Dev = 2.93%
#         Gradient Boosting: Mean Accuracy = 82.49% | Std Dev = 3.42%
#       Logistic Regression: Mean Accuracy = 79.34% | Std Dev = 5.10%
# Support Vector Classifier: Mean Accuracy = 82.49% | Std Dev = 4.24%
