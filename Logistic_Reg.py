#Titanic Kagle Project Logistic Regression
#Programmer: Nathaniel Gosdin

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Feature Engineering
# Extract titles
def get_title(name):
    title = name.split(',')[1].split('.')[0].strip()
    # Group rare titles
    if title in ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']:
        return 'Rare'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title == 'Mme':
        return 'Mrs'
    else:
        return title

# Apply to both datasets
for dataset in [train, test]:
    dataset['Title'] = dataset['Name'].apply(get_title)
    
    # Create family size feature
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1
    dataset.loc[dataset['FamilySize'] > 1, 'IsAlone'] = 0
    
    # Extract deck from cabin
    dataset['Deck'] = dataset['Cabin'].apply(lambda x: str(x)[0] if pd.notna(x) else 'U')
    
    # Create age*class interaction
    dataset['Age*Class'] = dataset['Age'] * dataset['Pclass']
    
    # Fare per person
    dataset['FarePerPerson'] = dataset['Fare'] / dataset['FamilySize']

# Handle missing values
# Fill missing embarked with most common
for dataset in [train, test]:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    
# Convert categorical to numerical
for dataset in [train, test]:
    dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1})
    dataset['Title'] = dataset['Title'].map({'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5})
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    dataset['Deck'] = dataset['Deck'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'U': 0})

# Fill missing age values directly
age_imputer = SimpleImputer(strategy='median')
for dataset in [train, test]:
    dataset['Age'] = age_imputer.fit_transform(dataset[['Age']])

# Fill missing Fare in test set (avoid inplace)
if test['Fare'].isnull().any():
    median_fare = test.groupby('Pclass')['Fare'].transform('median')
    test['Fare'] = test['Fare'].fillna(median_fare)

# Recalculate Age*Class after imputation
for dataset in [train, test]:
    dataset['Age*Class'] = dataset['Age'] * dataset['Pclass']
    dataset['FarePerPerson'] = dataset['Fare'] / dataset['FamilySize']

# Select features for the model
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
           'Title', 'FamilySize', 'IsAlone', 'Age*Class', 'FarePerPerson']

# Prepare data for modeling
X_train = train[features]
y_train = train['Survived']
X_test = test[features]

# Double-check for any remaining NaN values
print("NaN values in X_train:")
print(X_train.isnull().sum())
# Impute any remaining NaN values
imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Final check for NaN values
print("NaN values after scaling:")
print(np.isnan(X_train_scaled).sum())

# Simplified Logistic Regression with just a few parameters to test
print("\nTraining Logistic Regression model:")
log_reg = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

# Calculate CV score to see performance
kfold = KFold(n_splits=10, random_state=42, shuffle=True)
cv_scores = cross_val_score(log_reg, X_train_scaled, y_train, cv=kfold)
print(f"{'Logistic Regression':>25}: Mean Accuracy = {cv_scores.mean():.2%} | Std Dev = {cv_scores.std():.2%}")

# Make predictions on training set for confusion matrix
train_predictions = log_reg.predict(X_train_scaled)

# Create and display confusion matrix
cm = confusion_matrix(y_train, train_predictions)
plt.figure(figsize=(8, 6))
labels = ['Not Survived', 'Survived']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix for Training Data', fontsize=14)
plt.tight_layout()
plt.show()

# Display classification report
print("\nClassification Report:")
print(classification_report(y_train, train_predictions, target_names=labels))

# Make predictions on test set
log_reg_pred = log_reg.predict(X_test_scaled)

# Create submission file
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': log_reg_pred
})
submission.to_csv('logistic_regression_improved.csv', index=False)
print("Submission file created: logistic_regression_improved.csv")

# Analyze coefficients to understand feature importance
if hasattr(log_reg, 'coef_'):
    coefficients = pd.DataFrame({
        'Feature': features,
        'Coefficient': log_reg.coef_[0]
    }).sort_values(by='Coefficient', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(coefficients['Feature'], coefficients['Coefficient'])
    plt.title('Logistic Regression Coefficients')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()