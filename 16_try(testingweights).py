import pandas as pd
import numpy as np
from ann_ga import network_structure as ns
from ann_ga import data_retrieval as d_r

def retrive_details(name):
    layer_details=list((pd.read_csv(name,usecols=["layer"])).squeeze())
    weights=(pd.read_csv(name,usecols=["weights"])).squeeze()
    bias_list=list((pd.read_csv('file1.csv',usecols=["bias"])).squeeze())
    weights=np.array(weights)
    weights=list(weights)

    weight_list=[]
    for x in weights:
        y=x.replace('[','')
        y=y.replace(']','')
        y=y.replace(',','')
        y=y.split(' ')
        for i in range(len(y)):
            y[i]=float(y[i])
        weight_list.append(y)

    h_count,o_count=0,0
    for x in layer_details:
        if(x == 'h'): h_count+=1
        if(x == 'w'): o_count+=1
    return h_count,o_count,bias_list,weight_list

name=input('Enter the file name of the weight_bias csv: ')
h_count,o_count,bias_list,weight_list=retrive_details(name)

net_obj=ns.neural_network()
net_obj.existing_structure_definition(len(weight_list[0]),o_count,10,weight_list[0:h_count],weight_list[h_count:],bias_list[0:h_count],bias_list[h_count:])
file_name=input('Enter file name of the dataset: ')
L=d_r.retrieve_data(file_name)
mydata=pd.read_csv(file_name)
input_parameters_names,input_parameters=[],[]
for i in range(len(mydata.columns)):
    if(i != L[0]): input_parameters_names.append(mydata.columns[i])
for x in input_parameters_names:
    input_parameters.append(float(input(f'Enter {x}: ')))
net_obj.forward_pass(input_parameters)
m,max_index=-9999,-1
for i in range(len(net_obj.o)):
    if(net_obj.o[i].activation_value > 0.5):
        if(net_obj.o[i].activation_value == m):
            max_index=-1
        if(net_obj.o[i].activation_value > m):
            m=net_obj.o[i].activation_value
            max_index=i

if(max_index != -1):
    print(f'\n\nOBJECT DETECTED: {L[4][max_index]}')