#!/usr/bin/env python
# coding: utf-8

# In[33]:


# !pip3 install pandas
# !pip3 install matplotlib
# !pip3 install seaborn
# !pip3 install sklearn


# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score


# In[35]:


data = pd.read_csv('bc_data.csv');
print("\n \t The data frame has {0[0]} rows and {0[1]} columns. \n".format(data.shape))
# data.info()

data.head(3)


# In[36]:


data.drop(data.columns[[-1, 0]], axis=1, inplace=True) #removing the extra column created when uploading data set

data.info()


# In[37]:


# diagnosis_all = list(data.shape)[0]
# diagnosis_categories = list(data['diagnosis'].value_counts())

# print("\n \t The data has {} diagnosis, {} malignant and {} benign.".format(diagnosis_all, diagnosis_categories[0], diagnosis_categories[1]))


# In[38]:


features_mean= list(data.columns[1:11])


# In[39]:


# plt.figure(figsize=(10,10))
# sns.heatmap(data[features_mean].corr(), annot=True, square=True, cmap='coolwarm')
# plt.show()


# In[40]:


# color_dic = {'M':'red', 'B':'blue'}
# colors = data['diagnosis'].map(lambda x: color_dic.get(x))

# sm = pd.scatter_matrix(data[features_mean], c=colors, alpha=0.4, figsize=((15,15)));

# plt.show()


# In[41]:


# bins = 12
# plt.figure(figsize=(15,15))
# for i, feature in enumerate(features_mean):
#     rows = int(len(features_mean)/2)
    
#     plt.subplot(rows, 2, i+1)
    
#     sns.distplot(data[data['diagnosis']=='M'][feature], bins=bins, color='red', label='M');
#     sns.distplot(data[data['diagnosis']=='B'][feature], bins=bins, color='blue', label='B');
    
#     plt.legend(loc='upper right')

# plt.tight_layout()
# plt.show()


# In[42]:


# plt.figure(figsize=(15,15))
# for i, feature in enumerate(features_mean):
#     rows = int(len(features_mean)/2)
    
#     plt.subplot(rows, 2, i+1)
    
#     sns.boxplot(x='diagnosis', y=feature, data=data, palette="Set1")

# plt.tight_layout()
# plt.show()


# In[43]:


# features_selection = ['radius_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 'concave points_mean']


# In[44]:


diag_map = {'M':1, 'B':0}
data['diagnosis'] = data['diagnosis'].map(diag_map)


# In[45]:


X = data.loc[:,features_mean]
y = data.loc[:, 'diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

accuracy_all = []
cvs_all = []
time_all = []


# ## Decision Trees

# #### Using Scikit-learn's Decision Tree algorithm

# In[46]:


from sklearn.tree import DecisionTreeClassifier


# In[47]:


start = time.time()

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))
time_all.append(end-start)

print("Without Pruning:")
print("Decision Tree Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))


# In[48]:


start = time.time()

clf = DecisionTreeClassifier(max_leaf_nodes = 10, min_samples_leaf = 3, max_depth= 5)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))
time_all.append(end-start)

print("With Pruning:")
print("Decision Tree Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))


# Pruning helped to avoid overfitting, therefore resulting in a better accuracy and cross validation score. I chose to prune by setting the following parameters:
# 
# max_leaf_nodes = 10 (default is none) in attempt to reduce the number of leaf nodes that would be unnecessary
# 
# min_samples_leaf = 5 (default is 1) in attempt to restrict the size of sample leaf
# 
# max_depth = 5 (default is none) in attempt to reduce the depth of the tree to build a more generalized tree

# ## Neural Networks

# #### Using Scikit-learn's Neural Network Multi-layer Perceptron classifier 

# In[49]:


from sklearn.neural_network import MLPClassifier


# In[99]:


start = time.time()

clf = MLPClassifier(max_iter=200)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))
time_all.append(end-start)

print("Neural Networks Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))


# ## Boosting

# In[51]:


from sklearn.ensemble import GradientBoostingClassifier


# In[52]:


start = time.time()

clf = GradientBoostingClassifier(n_estimators = 200, max_depth = 1)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))
time_all.append(end-start)

print("Boosting Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))


# Pruning helped to avoid overfitting, therefore resulting in a better accuracy and cross validation score. I chose to prune by setting the following parameters:
# 
# n_estimators = 200 (default is 100) "The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance."
# 
# max_depth = 1 (default is 3) in attempt to reduce the depth of the tree to build a more generalized tree

# ## Support Vector Machines (Linear)

# #### Using Scikit-learn's SVM algorithm

# In[53]:


from sklearn.svm import SVC


# In[54]:


start = time.time()

clf = SVC(kernel ='linear')
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))
time_all.append(end-start)

print("Linear SVM Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))


# ## k-Nearest Neighbors

# #### Using Scikit-learn's kNN algorithm

# In[55]:


from sklearn.neighbors import KNeighborsClassifier


# In[56]:


start = time.time()

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))
time_all.append(end-start)

print("n_neighbors = 5:")
print("Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))


# In[57]:


start = time.time()

clf = KNeighborsClassifier(n_neighbors=2)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))
time_all.append(end-start)

print("n_neighbors = 2:")
print("Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))


# In[58]:


start = time.time()

clf = KNeighborsClassifier(n_neighbors=20)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))
time_all.append(end-start)

print("n_neighbors = 20:")
print("Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))


# Lowering the number of neighbors from the default of 5 lowered the accuracy. However, increasing the number of neighbors did not increase the accuracy of classification by much.

# ## Comparison of Algorithms

# In[59]:


d = {'accuracy':accuracy_all,'cross validation score':cvs_all, 'execution time (seconds)':time_all}

index = ['Decision Trees without Pruning', 'Decision Trees with Pruning', 'Neural Networks', 
         'Boosting', 'Support Vector Machines', '5-Nearest Neighbors', '2-Nearest Neighbors', '20-Nearest Neighbors']

df = pd.DataFrame(d, index=index)

df


# In[63]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:        
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[102]:


title = "Decision Tree Classifier without pruning"
cv = 5
estimator = DecisionTreeClassifier()
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

title = "Decision Tree Classifier with pruning"
cv = 5
estimator = DecisionTreeClassifier(max_leaf_nodes = 10, min_samples_leaf = 3, max_depth= 5)
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

# title = "Neural Networks"
# cv = 5
# estimator = MLPClassifier(hidden_layer_sizes=(10))
# plot_learning_curve(estimator, title, X, y, ylim=(0.3, 1.01), cv=cv, n_jobs=4)

title = "Boosting"
cv = 5
estimator = GradientBoostingClassifier(n_estimators = 200, max_depth = 1)
plot_learning_curve(estimator, title, X, y, ylim=(0.5, 1.01), cv=cv, n_jobs=4)

title = "Support Vector Machines"
cv = 5
estimator = SVC(kernel ='linear')
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

title = "5-Nearest Neighbors"
cv = 5
estimator = KNeighborsClassifier()
plot_learning_curve(estimator, title, X, y, ylim=(0.4, 1.01), cv=cv, n_jobs=4)

title = "2-Nearest Neighbors"
cv = 5
estimator = KNeighborsClassifier(n_neighbors=2)
plot_learning_curve(estimator, title, X, y, ylim=(0.5, 1.01), cv=cv, n_jobs=4)

title = "20-Nearest Neighbors"
cv = 5
estimator = KNeighborsClassifier(n_neighbors=20)
plot_learning_curve(estimator, title, X, y, ylim=(0.3, 1.01), cv=cv, n_jobs=4)

plt.show()


# In[83]:


## seperate run for neural networks bc slow

title = "Neural Networks"
cv = 5
estimator = MLPClassifier(hidden_layer_sizes=(10, ))
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
plt.show()


# In[ ]:





# In[ ]:




