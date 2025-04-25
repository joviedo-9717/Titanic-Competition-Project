#Titanic Kagle Project Enhanced Support Vector Classifier
#Programmer: Nathaniel Gosdin

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Simple feature engineering - only the most important features
for dataset in [train, test]:
    # Extract titles
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Group rare titles
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 
                                                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
    # Create family size
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
    # Is alone indicator
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Fill missing values
for dataset in [train, test]:
    # Fill missing embarked with most common value
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    
    # Convert categorical to numerical
    dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1})
    title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Fill missing age values with median by title and pclass
for dataset in [train, test]:
    for title in dataset['Title'].unique():
        title_age_median = dataset[dataset['Title'] == title]['Age'].median()
        if not np.isnan(title_age_median):
            dataset.loc[(dataset['Age'].isnull()) & (dataset['Title'] == title), 'Age'] = title_age_median
    
    # Any remaining missing ages with global median
    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())

# Fill missing fare in test set
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

# Create child indicator
for dataset in [train, test]:
    dataset['IsChild'] = 0
    dataset.loc[dataset['Age'] < 16, 'IsChild'] = 1

# Select features
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone', 'IsChild']

# Prepare data
X_train = train[features]
y_train = train['Survived']
X_test = test[features]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVC with optimal parameters
print("Training Support Vector Classifier...")
svc = SVC(C=5.0, kernel='rbf', gamma=0.1, random_state=42)
svc.fit(X_train_scaled, y_train)

# Check training accuracy
train_pred = svc.predict(X_train_scaled)
train_accuracy = (train_pred == y_train).mean()
print(f"Training Accuracy: {train_accuracy:.4f}")

# Make predictions
test_pred = svc.predict(X_test_scaled)

# Create submission file
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': test_pred
})
submission.to_csv('svc_simple.csv', index=False)
print("Submission file created: svc_simple.csv")