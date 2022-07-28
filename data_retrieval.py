import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def retrieve_data(file_name):
    mydata=pd.read_csv(file_name) 
    print(mydata)
    print(mydata.nunique())
    output_columnnum=int(input('\nEnter the column number containing the classes: '))
    Yt=pd.read_csv(file_name,usecols=[output_columnnum])
    input_columnnum=[x for x in range(len(mydata.loc[0])) if x != output_columnnum]
    Xt=pd.read_csv(file_name,usecols=input_columnnum)
    return parameter_identification(Xt,Yt)

def parameter_identification(Xt,Yt):
    output_cases=list((Yt.squeeze()).unique())
    X_train,x_test,Y_train,y_test=train_test_split(np.array(Xt),np.array(Yt.squeeze()),test_size=0.1,random_state=3,shuffle=True)
    parameters=len(Xt.loc[0])
    Xt=list(X_train)
    for i in range(len(Xt)):
        Xt[i]=list(Xt[i])
    x_test=list(x_test)
    for i in range(len(x_test)):
        x_test[i]=list(x_test[i])
    for x in x_test:
        Xt.append(x)
    Yt=list(Y_train)
    y_test=list(y_test)
    Yt=Yt+y_test[:]
    output_classes=len(set(Yt))
    return [parameters,output_classes,Xt,Yt,output_cases]