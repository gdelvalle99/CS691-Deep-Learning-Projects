import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import svm


N = 250
size = int(N/2)
Uh = 20
Ul = -1
x_min2 = [Ul, Ul]
x_max2 = [-Uh, Uh]
x_min4 = [Ul, -Ul]
x_max4 = [Uh, -Uh]

o_min1 = [Ul, Ul]
o_max1 = [Uh, Uh]
o_min3 = [-Ul, -Ul]
o_max3 = [-Uh, -Uh]



O1 = np.random.uniform(low=o_min1, high=o_max1, size=(size,2))
X2 = np.random.uniform(low=x_min2, high=x_max2, size=(size,2))
O3 = np.random.uniform(low=o_min3, high=o_max3, size=(size,2))
X4 = np.random.uniform(low=x_min4, high=x_max4, size=(size,2))
O = np.concatenate((O1,O3), axis=0)
X = np.concatenate((X2,X4), axis=0)

x_train = np.concatenate((O,X), axis=0)


y_train = None

for index, row in enumerate(x_train):
    if( index < N):
        y_o = np.array([0,1])
        if(index == 0):
            y_train = y_o
        else:
            y_train = np.vstack((y_train,y_o))
    else:
        y_x = np.array([1,0])
        y_train = np.vstack((y_train,y_x))

#y_train = np.vstack((y_o,y_x))
#print(y_train.shape)

model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(2,)))
model.add(Dense(2, activation='sigmoid'))
model.summary()
print(model.get_config())
print(model.get_weights())

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])



#plt.scatter(X[:,0],X[:,1],marker='+',c='blue', label='X-class')
#plt.scatter(O[:,0],O[:,1],marker='o',c='red', edgecolors='none', label='O-class')
#plt.legend()
#plt.grid(True)
#plt.show()
epochs = 100
n = 50

history = model.fit(x=x_train,y=y_train,batch_size=n,epochs=epochs,verbose=1)


NTest = 150
sizeTest = int(NTest/2)
#print(sizeTest)
O1test = np.random.uniform(low=o_min1, high=o_max1, size=(sizeTest,2))
X2test = np.random.uniform(low=x_min2, high=x_max2, size=(sizeTest,2))
O3test = np.random.uniform(low=o_min3, high=o_max3, size=(sizeTest,2))
X4test = np.random.uniform(low=x_min4, high=x_max4, size=(sizeTest,2))
Otest = np.concatenate((O1test,O3test), axis=0)
Xtest = np.concatenate((X2test,X4test), axis=0)

x_test = np.vstack((Otest,Xtest))
#print(x_test.shape)
#plt.scatter(Xtest[:,0],Xtest[:,1],marker='+',c='blue', label='X-class')
#plt.scatter(Otest[:,0],Otest[:,1],marker='o',c='red', edgecolors='none', label='O-class')
#plt.legend()
#plt.grid(True)
#plt.show()

y_test = None

for index, row in enumerate(x_test):
    if( index < NTest):
        y_o = np.array([0,1])
        if(index == 0):
            y_test = y_o
        else:
            y_test = np.vstack((y_test,y_o))
    else:
        y_x = np.array([1,0])
        y_test = np.vstack((y_test,y_x))
    #print(x_test[index], y_test[index])


score = model.evaluate(x=x_test,y=y_test)
print(y_test.shape)
false_positives = None
false_negatives = None
X_true = None
O_true = None
q = model.predict(x_test)
#we assume that X is positive and O is negative
for index, row in enumerate(q):
    #print(q[0],q[1],y_test[index][0])
    if((row[0] > row[1] and y_test[index][0] > y_test[index][1]) or(row[1] > row[0] and y_test[index][1] > y_test[index][0])):
        #print(y_test[index])
        if(row[0] > row[1]):
            if(X_true is None):
                X_true = x_test[index]
            else:
                #print('here')
                X_true = np.vstack((X_true, x_test[index]))
        else:
            if(O_true is None):
                O_true = x_test[index]
            else:
                #print(x_test[index])
                O_true = np.vstack((O_true, x_test[index]))
    elif((row[0] > row[1] and y_test[index][1] > y_test[index][0])):#false positive
        if(false_positives is None):
            false_positives = x_test[index]
        else:
            false_positives = np.vstack((false_positives, x_test[index]))
    elif((row[1] > row[0] and y_test[index][0] > y_test[index][1])):
        if(false_negatives is None):
            false_negatives = x_test[index]
        else:
            false_negatives = np.vstack((false_negatives, x_test[index]))

v_line = np.concatenate((O1test,X2test),axis=0)
h_line = np.concatenate((O3test,X2test),axis=0)
y_h = None
y_v = None
q_h = model.predict(h_line)
q_v = model.predict(v_line)

for index, i in enumerate(q_h):
    if(q_h[index][1] > q_h[index][0]):
        if y_h is None:
            y_h = 0
        else:
            y_h = np.hstack((y_h,0))
    elif(q_h[index][0] > q_h[index][1]):
        if y_h is None:
            y_h = 1
        else:
            y_h = np.hstack((y_h,1))

for index, i in enumerate(q_v):
    if(q_v[index][1] > q_v[index][0]):
        if y_v is None:
            y_v = 0
        else:
            y_v = np.hstack((y_v,0))
    elif(q_v[index][0] > q_v[index][1]):
        if y_v is None:
            y_v = 1
        else:
            y_v = np.hstack((y_v,1))


C = 1.0  # SVM regularization parameter
clf_v = svm.SVC(kernel = 'linear',  gamma=0.7, C=C )
clf_v.fit(v_line, y_v)

clf_h = svm.SVC(kernel = 'linear',  gamma=0.7, C=C )
clf_h.fit(h_line, y_h)

w_h = clf_h.coef_[0]
a_h = -w_h[0] / w_h[1]
xx_h = np.linspace(-20, 20)
yy_h = a_h * xx_h - (clf_h.intercept_[0]) / w_h[1]

plt.plot(xx_h, yy_h, 'k-')

w_v = clf_v.coef_[0]
a_v = -w_v[0] / w_v[1]
xx_v = np.linspace(-5, 5)
yy_v = a_v * xx_v - (clf_v.intercept_[0]) / w_v[1]

plt.plot(xx_v, yy_v, 'k-')

#plt.plot(dec_bound[:,0], dec_bound[:,1])
#print(X_true)
#print(score)
if(X_true is not None):
    plt.scatter(X_true[:,0],X_true[:,1],marker='+',c='blue', label='X-class')
if(O_true is not None):
    plt.scatter(O_true[:,0],O_true[:,1],marker='o',c='red', edgecolors='none', label='O-class')
if(false_positives is not None):
    plt.scatter(false_positives[:,0],false_positives[:,1],marker='+',c='yellow', label='False positives')
if(false_negatives is not None):
    plt.scatter(false_negatives[:,0],false_negatives[:,1],marker='o',c='green', edgecolors='none', label='False negatives')
plt.legend()
plt.grid(True)
plt.show()
