#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from numpy import *
from pandas import *
from matplotlib import *
import math
import csv
import sklearn
from sklearn import *
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV

import statistics

import os
from os.path import exists
import pickle

from datetime import datetime
from datetime import date
from datetime import time


# In[2]:


x = pd.read_csv(r'SetOfFeaturesLR.csv')
x = x.set_index('Date')
y = pd.read_csv(r'values.csv')
y = y.set_index('Date')


# In[3]:


x.head()


# In[4]:


x.describe()


# In[5]:


y.describe()


# In[6]:


Combine=pd.concat([x,y],axis=1)
Combine.head()


# In[7]:


Combine.corr(method ='pearson')


# In[8]:


y['outcome'].value_counts(sort=False, ascending=True)


# In[10]:


#Initial weights
w=np.array([1,2,3,4,5,6,7,8])

#Probability of success 
def sigmoid(m,w): 
    return 1.0/(1 + np.exp(-np.dot(m,w)))
def compute_cost(y,m,w): #Scalar
    h=sigmoid(m,w)
    arg= (y*np.log(h)+(1-y)*np.log(1-h)) 
    return -(arg.T.dot(arg))/len(y)
def compute_gradient(y,m,w): #procedure to find the gradient vector
    err = sigmoid(m,w)-y
    grad = (m.T.dot(err))/len(y) 
    return grad, err
def gradient_descent(y, m, initial_w, max_iters, tau):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, m, w)
        loss = compute_cost(y,m,w)
        # gradient descent update
        w = w - tau * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}, w2={w2}, w3={w3}, w4={w4}, w5={w5}, w6={w6}".format(
              bi=n_iter, ti=max_iters - 1, l=loss,  w0=w[0], w1=w[1], w2=w[2], w3=w[3], w4=w[4], w5=w[5], w6=w[6]))
    return losses, ws
def last_iter_parameters(y, m, initial_w, max_iters, tau):
    ws = [initial_w]
    losses = np.array([])
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, m, w)
        # gradient descent update
        w = w - tau * grad
        # store w and loss
        ws.append(w)
    return [w[0],w[1], w[2], w[3], w[4], w[5], w[6], w[7]] #gives a list of parameters for the last iterate
def losses(y, m, initial_w, max_iters, tau):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, m, w)
        loss = compute_cost(y,m,w)
        # gradient descent update
        w = w - tau * grad
        # store w and loss
        losses.append(loss)
        print (np.asarray(loss))
def lossesplot(y, m, initial_w, max_iters, tau):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, m, w)
        loss = compute_cost(y,m,w)
        # gradient descent update
        w = w - tau * grad
        # store w and loss
        losses.append(loss)
    cost = list(losses)
    n_iterations = [x for x in range(1,max_iters+1)]
    plt.plot(n_iterations, cost)
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost')
def predictions(X,w):
    preds=[]
    for i in sigmoid(X, w):
        if i> 0.5:
            preds.append(1)
        if i < 0.5:
            preds.append(0)
    return np.array(preds)
#Gives test accuracy for the parameters, i.e. how many predictions were correct
def accuracy(X,w,y):
    return len(np.array((np.where(predictions(X,w)==y))).T)/len(y)
def parameters_accuracies(y,x,tau,xt,yt):
    for i in range (500,5000,500):
        a= last_iter_parameters(y,x,w,i,tau)
        b=accuracy(xt,a,yt)
        print("with", i,"iterations and stepsize=", tau, "we get the following test accuracy:", b)


# In[11]:


Matrix=np.c_[np.ones((x.shape[0],1)),x]
Matrix, Matrix.shape


# In[12]:


Combine_matrix=np.array(pd.concat([x,y],axis=1))
Matrix_Combine=np.insert(Combine_matrix,0,np.array((np.ones(x.shape[0]))),1) #adding ones column so that we can take the matrix product
Matrix_Combine, Matrix_Combine.shape


# In[14]:

#parameter config 
outOfSample_startdate= 'str'
trainSize= int 
poly_degree= 'str'
ploy_interaction_only_TrueFalse= bool
LR_penaly_l1l2= 'none'
LR_c= 'str'

c = pd.DataFrame()  #intercept values
coefs = pd.DataFrame()     #coefs values

y_pred_rolling_LR = pd.DataFrame()
y_prob_rolling_LR = pd.DataFrame()

outOfSample_startdate = outOfSample_startdate
trainSize = trainSize
split = x.index.get_loc(outOfSample_startdate) 

y_test= Matrix_Combine[split:][:,-1]
    
for i in range(len(y_test)):
    split = x.index.get_loc(outOfSample_startdate) + i
    x_train, x_test, y_train = Matrix[split-trainSize:split], Matrix[split:], Matrix_Combine[split-trainSize:split][:,-1]
        
    poly = PolynomialFeatures(degree = poly_degree, interaction_only=ploy_interaction_only_TrueFalse, include_bias=True)
    #x_poly = poly.fit_transform(x_train)
    
    a= last_iter_parameters(y_train, x_train, w, 500, 1.5)
    y_pred= predictions(x_test,a).reshape(-1,1)
    y_pred_rolling_LR= y_pred_rolling_LR.append(pd.DataFrame(y_pred[0]))

    #LR = LogisticRegression(penalty=LR_penaly_l1l2, C=LR_c, solver='liblinear').fit(x_poly, y_train)
    #c = c.append(pd.DataFrame(LR.intercept_.reshape(1,-1)), ignore_index=True)
    #coefs = coefs.append(pd.DataFrame(LR.coef_.reshape(1,-1)), ignore_index=True)
    #y_pred_rolling_LR = y_pred_rolling_LR.append(pd.DataFrame(LR.predict(poly.transform(x_test)).reshape(1,-1)), ignore_index=True)
    #y_prob_rolling_LR = y_prob_rolling_LR.append(pd.DataFrame(LR.predict_proba(poly.transform(x_test)).reshape(1,-1)), ignore_index=True)
        
    print('x_train :')
    print(x_train)

    print('x_test :')
    print(x_test)


# In[15]:


print('Confusion Matrix of Logistic Regression:')
print(metrics.confusion_matrix(y_test, y_pred_rolling_LR))

print('Classification report of Logistic Regression:')
print(metrics.classification_report(y_test, y_pred_rolling_LR))


LRClassificationReport = metrics.classification_report(y_test, y_pred_rolling_LR)
LRClassificationReportDict = metrics.classification_report(y_test, y_pred_rolling_LR, output_dict = True)
LRConfusionMatrix = metrics.confusion_matrix(y_test, y_pred_rolling_LR)


# In[16]:


last_iter_parameters(y_train, x_train, w, 500, 1.5)


# In[17]:


lossesplot(y_train, x_train, w, 500, 1.5)


# In[22]:


losses(y_train, x_train, w, 500, 1.5)


# In[18]:


for i in np.arange(0.1,2,0.1):
    print(parameters_accuracies(y_train,x_train,i,x_test,y_test)) #choose gamma=1.9, max_iters=500


# In[19]:


parameters_accuracies(y_train,x_train,1.3,x_test,y_test)


# In[20]:


last_iter_parameters(y_train, x_train, w, 500, 1.3)


# In[21]:


accuracy(x_train,last_iter_parameters(y_train, x_train, w, 500, 1.3),y_train)


# In[ ]:


#example: LRfit('2015-06-03', 1000, 2, False, 'l2', 10)  
outOfSample_startdate= '2019-06-03'
trainSize= 1000
poly_degree= 2
ploy_interaction_only_TrueFalse= False
LR_penaly_l1l2= 'l2'
LR_c= 10

c = pd.DataFrame()  #intercept values
coefs = pd.DataFrame()     #coefs values

y_pred_rolling_LR = pd.DataFrame()
y_prob_rolling_LR = pd.DataFrame()

outOfSample_startdate = outOfSample_startdate
trainSize = trainSize
split = x.index.get_loc(outOfSample_startdate) 

y_test= y[split:]
    
for i in range(len(y_test)):
    split = x.index.get_loc(outOfSample_startdate) + i
    x_train, x_test, y_train = x[split-trainSize:split], x[split:], y[split-trainSize:split]
        
    poly = PolynomialFeatures(degree = poly_degree, interaction_only=ploy_interaction_only_TrueFalse, include_bias=True)
    x_poly = poly.fit_transform(x_train)

    LR = LogisticRegression(penalty=LR_penaly_l1l2, C=LR_c, solver='liblinear').fit(x_poly, y_train)
    c = c.append(pd.DataFrame(LR.intercept_.reshape(1,-1)), ignore_index=True)
    coefs = coefs.append(pd.DataFrame(LR.coef_.reshape(1,-1)), ignore_index=True)
    y_pred_rolling_LR = y_pred_rolling_LR.append(pd.DataFrame(LR.predict(poly.transform(x_test)).reshape(1,-1)), ignore_index=True)
    y_prob_rolling_LR = y_prob_rolling_LR.append(pd.DataFrame(LR.predict_proba(poly.transform(x_test)).reshape(1,-1)), ignore_index=True)
        
    print('x_train :')
    print(x_train)

    print('x_test :')
    print(x_test)


# In[ ]:


y_pred_rolling_LR = pd.DataFrame(y_pred_rolling_LR.iloc[:,0])
y_pred_rolling_LR.columns=['y_pred_LR']

y_prob_rolling_LR = pd.DataFrame(y_prob_rolling_LR.iloc[:,0]) #probability of 0
colnames=['prob0']
y_prob_rolling_LR.columns = colnames

y_prob_rolling_LR['prob1'] = 1 - y_prob_rolling_LR['prob0']

y_pred_rolling = pd.merge(y_pred_rolling_LR, y_prob_rolling_LR, left_index=True, right_index=True)

#y_pred_rolling.to_csv('y_pred_rolling_062019.csv')


# In[ ]:


print('Confusion Matrix of Logistic Regression:')
print(metrics.confusion_matrix(y_test, y_pred_rolling_LR))

print('Classification report of Logistic Regression:')
print(metrics.classification_report(y_test, y_pred_rolling_LR))


LRClassificationReport = metrics.classification_report(y_test, y_pred_rolling_LR)
LRClassificationReportDict = metrics.classification_report(y_test, y_pred_rolling_LR, output_dict = True)
LRConfusionMatrix = metrics.confusion_matrix(y_test, y_pred_rolling_LR)




