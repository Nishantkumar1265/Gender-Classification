# Gender-Classification
!pip install numpy

!pip install pandas




import numpy as np


import pandas as pd
data = pd.read_csv('/content/drive/MyDrive/gender_classification.csv')
data.head()

data.info

data.describe()

data

data["gender"].value_counts()

target_column = 'gender'
# Separate features (X) and target variable (y)
X = data.drop(target_column, axis=1)
y = data[target_column]

!pip install scikit-learn


# Decision tree model

from sklearn.model_selection import train_test_split
# Optionally, split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of X_train", X_train.shape)
print("Shape of X_test", X_test.shape)
print("Shape of y_train", y_train.shape)
print("Shape of y_test", y_test.shape)


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()




clf.fit(X_train, y_train)



print("Training accuracy: ", clf.score(X_train, y_train))
print("Test accuracy: ", clf.score(X_test, y_test))

from sklearn.metrics import confusion_matrix

print("Confusion matrix for training: \n",confusion_matrix(y_train, clf.predict(X_train)),'\n')
print("Confusion matrix for testing: \n",confusion_matrix(y_test, clf.predict(X_test)))

# K-Nearest Neighbour

from sklearn.neighbors import KNeighborsClassifier
clf1=KNeighborsClassifier()
clf1.fit(X_train, y_train)



print("Training accuracy: ", clf1.score(X_train, y_train))
print("Test accuracy: ", clf1.score(X_test, y_test))

print("Confusion matrix for training: \n",confusion_matrix(y_train, clf1.predict(X_train)),'\n')
print("Confusion matrix for testing: \n",confusion_matrix(y_test, clf1.predict(X_test)))

# Naive Bayes Classifier

from sklearn.naive_bayes import GaussianNB
clf2=GaussianNB()
clf2.fit(X_train, y_train)



print("Training accuracy: ", clf2.score(X_train, y_train))
print("Test accuracy: ", clf2.score(X_test, y_test))

print("Confusion matrix for training: \n",confusion_matrix(y_train, clf2.predict(X_train)),'\n')
print("Confusion matrix for testing: \n",confusion_matrix(y_test, clf2.predict(X_test)))

# logistic regression


from sklearn.linear_model import LogisticRegression
clf3 = LogisticRegression()
clf3.fit(X_train, y_train)




print("Training accuracy: ", clf3.score(X_train, y_train))
print("Test accuracy: ", clf3.score(X_test, y_test))

print("Confusion matrix for training: \n", confusion_matrix(y_train, clf3.predict(X_train)), '\n')
print("Confusion matrix for testing: \n", confusion_matrix(y_test, clf3.predict(X_test)))

# Classification report


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef


# List of classifiers
classifiers = ['Logistic Regression', 'k-Nearest Neighbors', 'Naive Bayes', 'Decision Trees']
# List of trained classifiers
models = [clf, clf1, clf2,clf3]

# Iterate through each model
for classifier, model in zip(classifiers, models):
# Make predictions
    y_pred = model.predict(X_test)
  # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

  # Print evaluation metrics
    print("Metrics for", classifier)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Cohen's Kappa:", kappa)
    print("Matthews Correlation Coefficient:", mcc)
    print()


ACCURACY COMPARISION

import matplotlib.pyplot as plt

# List of classifiers
classifiers = ['Logistic Regression', 'k-Nearest Neighbors', 'Naive Bayes', 'Decision Trees']

# List of accuracy scores for each classifier
accuracy_scores = [accuracy_score(y_test, clf.predict(X_test)),
                   accuracy_score(y_test, clf1.predict(X_test)),
                   accuracy_score(y_test, clf2.predict(X_test)),
                   accuracy_score(y_test, clf3.predict(X_test))]

# Plotting the bar graph for accuracy comparison
plt.figure(figsize=(10, 6))
bars = plt.bar(classifiers, accuracy_scores, color=['blue'])
plt.title('Accuracy Comparison')
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)

# Adding the values above the bars
for bar, score in zip(bars, accuracy_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), round(score, 5), ha='center', va='bottom')

plt.tight_layout()
plt.show()


# PRECISION COMPARISION

import matplotlib.pyplot as plt

# List of classifiers
classifiers = ['Logistic Regression', 'k-Nearest Neighbors', 'Naive Bayes', 'Decision Trees']

# List of precision scores for each classifier
precision_scores = [precision_score(y_test, clf.predict(X_test), average='weighted'),
                    precision_score(y_test, clf1.predict(X_test), average='weighted'),
                    precision_score(y_test, clf2.predict(X_test), average='weighted'),
                    precision_score(y_test, clf3.predict(X_test), average='weighted')]

# Plotting the bar graph for precision comparison
plt.figure(figsize=(10, 6))
bars = plt.bar(classifiers, precision_scores, color=[ 'gray'])
plt.title('Precision Comparison')
plt.xlabel('Classifiers')
plt.ylabel('Precision')
plt.xticks(rotation=45)

# Adding the values above the bars
for bar, score in zip(bars, precision_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), round(score, 5), ha='center', va='bottom')

plt.tight_layout()
plt.show()


# Recall Comparison

import matplotlib.pyplot as plt

# List of classifiers
classifiers = ['Logistic Regression', 'k-Nearest Neighbors', 'Naive Bayes', 'Decision Trees']

# List of recall scores for each classifier
recall_scores = [recall_score(y_test, clf.predict(X_test), average='weighted'),
                 recall_score(y_test, clf1.predict(X_test), average='weighted'),
                 recall_score(y_test, clf2.predict(X_test), average='weighted'),
                 recall_score(y_test, clf3.predict(X_test), average='weighted')]

# Plotting the bar graph for recall comparison
plt.figure(figsize=(10, 6))
bars = plt.bar(classifiers, recall_scores, color=['red'])
plt.title('Recall Comparison')
plt.xlabel('Classifiers')
plt.ylabel('Recall')
plt.xticks(rotation=45)

# Adding the values above the bars
for bar, score in zip(bars, recall_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), round(score, 5), ha='center', va='bottom')

plt.tight_layout()
plt.show()


# F1 Score Comparison

import matplotlib.pyplot as plt

# List of classifiers
classifiers = ['Logistic Regression', 'k-Nearest Neighbors', 'Naive Bayes', 'Decision Trees']

# List of F1 scores for each classifier
f1_scores = [f1_score(y_test, clf.predict(X_test), average='weighted'),
             f1_score(y_test, clf1.predict(X_test), average='weighted'),
             f1_score(y_test, clf2.predict(X_test), average='weighted'),
             f1_score(y_test, clf3.predict(X_test), average='weighted')]

# Plotting the bar graph for F1 score comparison
plt.figure(figsize=(10, 6))
bars = plt.bar(classifiers, f1_scores, color=['pink'])
plt.title('F1 Score Comparison')
plt.xlabel('Classifiers')
plt.ylabel('F1 Score')
plt.xticks(rotation=45)

# Adding the values above the bars
for bar, score in zip(bars, f1_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), round(score, 5), ha='center', va='bottom')

plt.tight_layout()
plt.show()


# KAPPA COMPARISION

import matplotlib.pyplot as plt

# List of classifiers
classifiers = ['Logistic Regression', 'k-Nearest Neighbors', 'Naive Bayes', 'Decision Trees']

# List of Cohen's kappa scores for each classifier
kappa_scores = [cohen_kappa_score(y_test, clf.predict(X_test)),
                cohen_kappa_score(y_test, clf1.predict(X_test)),
                cohen_kappa_score(y_test, clf2.predict(X_test)),
                cohen_kappa_score(y_test, clf3.predict(X_test))]

# Plotting the bar graph for Cohen's kappa comparison
plt.figure(figsize=(10, 6))
bars = plt.bar(classifiers, kappa_scores, color=['yellow'])
plt.title("Cohen's Kappa Comparison")
plt.xlabel('Classifiers')
plt.ylabel("Cohen's Kappa")
plt.xticks(rotation=45)

# Adding the values above the bars
for bar, score in zip(bars, kappa_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), round(score, 5), ha='center', va='bottom')

plt.tight_layout()
plt.show()


# MCC COMPARISION

import matplotlib.pyplot as plt

# List of classifiers
classifiers = ['Logistic Regression', 'k-Nearest Neighbors', 'Naive Bayes', 'Decision Trees']

# List of MCC scores for each classifier
mcc_scores = [matthews_corrcoef(y_test, clf.predict(X_test)),
              matthews_corrcoef(y_test, clf1.predict(X_test)),
              matthews_corrcoef(y_test, clf2.predict(X_test)),
              matthews_corrcoef(y_test, clf3.predict(X_test))]

# Plotting the bar graph for MCC comparison
plt.figure(figsize=(10, 6))
bars = plt.bar(classifiers, mcc_scores, color=['black'])
plt.title("Matthews Correlation Coefficient (MCC) Comparison")
plt.xlabel('Classifiers')
plt.ylabel("MCC")
plt.xticks(rotation=45)

# Adding the values above the bars
for bar, score in zip(bars, mcc_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), round(score, 5), ha='center', va='bottom')

plt.tight_layout()
plt.show()
