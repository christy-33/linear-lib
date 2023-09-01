#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy,math


# In[3]:


data=pd.read_csv("C:\\Users\\CHRISTY HARSHITHA\\Downloads\\linear_train.csv")
data.head()


# In[4]:


ratio=0.75
rows=data.shape[0]
train_size=int(rows*ratio)
train_set=data[0:train_size]
test_set=data[train_size:]


# In[5]:


data=data.sample(frac=1)
data


# In[6]:


x=train_set.drop(['label','sno'],axis=1).values
y=train_set['label'].values

x_test=test_set.drop(['label','sno'],axis=1).values
y_test=test_set['label'].values


# In[7]:


print(x)
print(x_test)
print(x_test.shape)


# In[8]:


print(y)
print(y_test)
print(y_test.shape)


# In[9]:


def z_score(x):
    
    mu=np.mean(x)
    sigma=np.std(x)
    x_norm=(x-mu)/sigma
    
    return (x_norm,mu,sigma)


# In[10]:


x_norm,x_mu,x_sigma=z_score(x)

print(f"x_sigma={x_sigma}")
print(f"x_mu={x_mu}")
print(f"x_norm={x_norm}")


# In[11]:


y_norm,y_mu,y_sigma=z_score(y)

print(f"y_sigma={y_sigma}")
print(f"y_mu={y_mu}")
print(f"y_norm={y_norm}")


# In[12]:


def predict(x,w,b):
    
    p=np.dot(x,w)+b
    
    return p


# In[13]:


w_init=np.ones((20,))

b_init=2


# In[14]:


x_trial=x[0,:]
print(w_init.shape)
print(x_trial.shape)



print(f" x_trial value: {x_trial}")
f_wb=predict(x_trial,w_init,b_init)
print(f"f_wb shape={f_wb.shape},f_wb value={f_wb}")
type(x_trial)
type(f_wb)


# In[15]:


def cost_calc(x,y,w,b):
    m=x.shape[0]
    cost=0
    for i in range(m):
        f_wb_i=np.dot(x[i],w)+b
        cost=(f_wb_i-y[i])**2 + cost
        
    cost=cost/(2*m)
    return cost


# In[16]:


total_cost1=cost_calc(x_norm,y_norm,w_init,b_init)
print(f"cost before iterations={total_cost1}")
type(total_cost1)


# In[17]:


def grad_calc(x,y,w,b):
    m,n=x.shape
    dj_dw=np.zeros((n,))
    dj_db=0
    
    for i in range(m):
        err=(np.dot(x[i],w)+b)-y[i]
        for j in range(n):
            dj_dw[j]=(err+dj_dw[j])*x[i,j]
            dj_db=err+dj_db
        dj_dw= dj_dw/m
        dj_db= dj_db/m
        
    return dj_dw, dj_db
    


# In[18]:


dj_dw1,dj_db1=grad_calc(x,y,w_init,b_init)
print(f"dj_dw1={dj_dw1}")
print(f"dj_db1={dj_db1}")


# In[19]:


def grad_descent(w_in,b_in,x,y,alpha,iter_count):
    j_his=[]
    
    m,n=x.shape
    for i in range(iter_count):
        dj_dw,dj_db=grad_calc(x,y,w_in,b_in)
        w_in=w_in-(alpha*dj_dw)
        b_in=b_in-(alpha*dj_db)
        
        if i<100000:
            cost=cost_calc(x,y,w_in,b_in)
            j_his.append(cost)
            
        if i % (iter_count // 10) == 0:
            print(f"iteration={i:4}   cost={float(j_his[-1])}")
    return w_in,b_in,j_his


# In[21]:


w0=np.ones((20,))
b0=1
iterations=6000
alpha = 1e-7

w_norm,b_norm,J_his=grad_descent(w0,b0,x_norm,y_norm,alpha,iterations)
print(f"w found by gradient descent={w_norm}")
print(f"b found by gradient descent={b_norm}")


# In[ ]:


m,_ = x.shape
for k in range(m):
    print(f"prediction: {np.dot(x_norm[k], w_norm) + b_norm},target value={y_norm[k]}")


# In[26]:


fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_his)
ax2.plot(100 + np.arange(len(J_his[100:])), J_his[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()


# In[28]:


w_norm1,b_norm1,J_his=grad_descent(w_norm,b_norm,x_norm,y_norm,1e-5,10000)
print(f"w found by gradient descent={w_norm1}")
print(f"b found by gradient descent={b_norm1}")


# In[33]:


m,_ = x.shape
for k in range(m):
    print(f"prediction: {np.dot(x_norm[k], w_norm1) + b_norm1},target value={y_norm[k]}")


# In[30]:


fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_his)
ax2.plot(100 + np.arange(len(J_his[100:])), J_his[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()


# In[31]:


w_norm2,b_norm2,J_his=grad_descent(w_norm1,b_norm1,x_norm,y_norm,1e-3,10000)
print(f"w found by gradient descent={w_norm1}")
print(f"b found by gradient descent={b_norm1}")


# In[32]:


m,_ = x.shape
for k in range(m):
    print(f"prediction: {np.dot(x_norm[k], w_norm2) + b_norm2},target value={y_norm[k]}")


# In[34]:


w_norm3,b_norm3,J_his=grad_descent(w_norm2,b_norm2,x_norm,y_norm,1e-3,10000)
print(f"w found by gradient descent={w_norm3}")
print(f"b found by gradient descent={b_norm3}")


# In[35]:


m,_ = x.shape
for k in range(m):
    print(f"prediction: {np.dot(x_norm[k], w_norm3) + b_norm3},target value={y_norm[k]}")


# In[36]:


w_norm4,b_norm4,J_his=grad_descent(w_norm3,b_norm3,x_norm,y_norm,5e-3,10000)
print(f"w found by gradient descent={w_norm4}")
print(f"b found by gradient descent={b_norm4}")


# In[37]:


m,_ = x.shape
for k in range(m):
    print(f"prediction: {np.dot(x_norm[k], w_norm4) + b_norm4},target value={y_norm[k]}")


# In[38]:


fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_his)
ax2.plot(100 + np.arange(len(J_his[100:])), J_his[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()


# In[39]:


w_norm5,b_norm5,J_his=grad_descent(w_norm4,b_norm4,x_norm,y_norm,1e-2,10000)
print(f"w found by gradient descent={w_norm5}")
print(f"b found by gradient descent={b_norm5}")


# In[40]:


m,_ = x.shape
for k in range(m):
    print(f"prediction: {np.dot(x_norm[k], w_norm4) + b_norm4},target value={y_norm[k]}")


# In[41]:


fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_his)
ax2.plot(100 + np.arange(len(J_his[100:])), J_his[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()


# In[42]:


w_norm6,b_norm6,J_his=grad_descent(w_norm5,b_norm5,x_norm,y_norm,1e-2,10000)
print(f"w found by gradient descent={w_norm6}")
print(f"b found by gradient descent={b_norm6}")


# In[43]:


m,_ = x.shape
for k in range(m):
    print(f"prediction: {np.dot(x_norm[k], w_norm6) + b_norm6},target value={y_norm[k]}")


# In[44]:


fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_his)
ax2.plot(100 + np.arange(len(J_his[100:])), J_his[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()


# In[45]:


w_norm7,b_norm7,J_his=grad_descent(w_norm6,b_norm6,x_norm,y_norm,1e-2,10000)
print(f"w found by gradient descent={w_norm6}")
print(f"b found by gradient descent={b_norm6}")


# In[46]:


m,_ = x.shape
for k in range(m):
    print(f"prediction: {np.dot(x_norm[k], w_norm7) + b_norm7},target value={y_norm[k]}")


# In[47]:


fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_his)
ax2.plot(100 + np.arange(len(J_his[100:])), J_his[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()


# In[48]:


w_norm8,b_norm8,J_his=grad_descent(w_norm7,b_norm7,x_norm,y_norm,1e-2,10000)
print(f"w found by gradient descent={w_norm8}")
print(f"b found by gradient descent={b_norm8}")


# In[50]:


j_his_all=[]
for i in J_his:
    j_his_all.append(i)
    


# In[51]:


print(j_his_all)


# In[52]:


w_norm9,b_norm9,J_his=grad_descent(w_norm8,b_norm8,x_norm,y_norm,1e-2,10000)
print(f"w found by gradient descent={w_norm9}")
print(f"b found by gradient descent={b_norm9}")


# In[53]:


for i in J_his:
    j_his_all.append(i)
    


# In[54]:


fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_his)
ax2.plot(100 + np.arange(len(j_his_all[100:])), j_his_all[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()


# In[55]:


w_norm10,b_norm10,J_his=grad_descent(w_norm9,b_norm9,x_norm,y_norm,1e-2,10000)
print(f"w found by gradient descent={w_norm10}")
print(f"b found by gradient descent={b_norm10}")


# In[56]:


for i in J_his:
    j_his_all.append(i)


# In[57]:


fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_his)
ax2.plot(100 + np.arange(len(j_his_all[100:])), j_his_all[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()


# In[25]:


w_norm10=[1.00521844, 0.99583351, 1.00311741, 1.00200358, 1.00107299, 1.00193615,
 1.00139204, 1.00233792, 1.00267516, 0.96004203, 0.99985787, 1.00201561,
 1.00171368, 1.02409322, 1.05605406, 1.00224096, 0.9817419,  1.00191249,
 1.00202935 ,0.98062197]
b_norm10=0.660573810763161


# In[26]:


w_norm11,b_norm11,J_his=grad_descent(w_norm10,b_norm10,x_norm,y_norm,1e-2,10000)
print(f"w found by gradient descent={w_norm11}")
print(f"b found by gradient descent={b_norm11}")


# In[38]:


w_norm12,b_norm12,J_his=grad_descent(w_norm11,b_norm11,x_norm,y_norm,1e-2,10000)
print(f"w found by gradient descent={w_norm12}")
print(f"b found by gradient descent={b_norm12}")


# In[40]:


w_norm13,b_norm13,J_his=grad_descent(w_norm12,b_norm12,x_norm,y_norm,1e-2,10000)
print(f"w found by gradient descent={w_norm13}")
print(f"b found by gradient descent={b_norm12}")


# In[41]:


w_norm14,b_norm14,J_his=grad_descent(w_norm13,b_norm13,x_norm,y_norm,1e-2,10000)
print(f"w found by gradient descent={w_norm14}")
print(f"b found by gradient descent={b_norm14}")


# In[42]:


w_norm15,b_norm15,J_his=grad_descent(w_norm14,b_norm14,x_norm,y_norm,1e-2,10000)
print(f"w found by gradient descent={w_norm15}")
print(f"b found by gradient descent={b_norm15}")


# In[43]:


w_norm16,b_norm16,J_his=grad_descent(w_norm15,b_norm15,x_norm,y_norm,1e-2,10000)
print(f"w found by gradient descent={w_norm16}")
print(f"b found by gradient descent={b_norm16}")


# In[20]:


w_norm16=[1.00753095, 0.99398716, 1.00449886, 1.00289145, 1.00154847, 1.00279414,
 1.00200891, 1.00337395, 1.00386063, 0.94233453, 0.99979488, 1.00290881,
 1.00247308, 1.03476991, 1.08089307, 1.00323402, 0.97365088, 1.00275999,
 1.00292864, 0.97203464]
b_norm16=0.5101581606269984


# In[ ]:


w_norm17,b_norm17,J_his=grad_descent(w_norm16,b_norm16,x_norm,y_norm,1e-2,10000)
print(f"w found by gradient descent={w_norm17}")
print(f"b found by gradient descent={b_norm17}")


# In[21]:


x_norm1,x_mu1,x_sigma1=z_score(x_test)

print(f"x_sigma={x_sigma1}")
print(f"x_mu={x_mu1}")
print(f"x_norm={x_norm1}")


# In[22]:


y_norm1,y_mu1,y_sigma1=z_score(y_test)

print(f"y_sigma={y_sigma1}")
print(f"y_mu={y_mu1}")
print(f"y_norm={y_norm1}")


# In[23]:


m,_ = x_test.shape
for k in range(m):
    pred_y_lt=[]
    pred_y=np.dot(x_norm1[k], w_norm16) + b_norm16
    pred_y_lt.append(pred_y)
   


# In[24]:


y_actual_mean=np.sum(y_norm1)/m
print(y_actual_mean)


# In[26]:


m,_ = x_test.shape
for k in range(m):
    RSS=0
    N=(y_norm1[k]-(np.dot(x_norm1[k], w_norm16) + b_norm16))**2
    RSS+=N


# In[27]:


for i in y_norm1:
    TSS=0
    n=(i-y_actual_mean)**2
    TSS+=n


# In[28]:


R2=1-(RSS/TSS)
print(R2)


# In[ ]:




