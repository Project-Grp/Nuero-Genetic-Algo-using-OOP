import numpy as np
import random
from sklearn.model_selection import train_test_split
import network_structure as ns

def data(Xt,Yt):
    X_train,x_test,Y_train,y_test=train_test_split(np.array(Xt),np.array(Yt),test_size=0.2,random_state=3,shuffle=True)
    return X_train,x_test

def back_prop(network_object,Xt,Yt):
    
    pass