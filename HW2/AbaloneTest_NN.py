#!/usr/bin/env python
# coding: utf-8

# In[3]:


# !pip3 install pandas
# !pip3 install matplotlib
# !pip3 install seaborn
# !pip3 install sklearn


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score


# In[8]:


data = pd.read_csv('abalone.txt', header=None);
print("\n \t The data frame has {0[0]} rows and {0[1]} columns. \n".format(data.shape))
# data.info()

data.head(3)


# In[12]:


data['8'] = np.where(data[7]>=15, 1, 0)
data.head(10)


# In[16]:


features_mean= list(data.columns[0:7])
features_mean


# In[17]:


X = data.loc[:,features_mean]
y = data.loc[:, '8']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

accuracy_all = []
cvs_all = []
time_all = []


# In[18]:


from sklearn.neural_network import MLPClassifier


# In[28]:


start = time.time()

clf = MLPClassifier(hidden_layer_sizes=(7,7,7,7,7), max_iter=2500, )
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


# In[29]:


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


# In[ ]:


title = "Neural Networks"
cv = 5
estimator = MLPClassifier(hidden_layer_sizes=(7,7,7,7,7), max_iter=2500, )
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
plt.show()


# In[ ]:




