import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

class neuron:

    def __init__(self,lNo):
        self.inputconnect=[]
        self.activation_value=0
        self.calculated_value=0
        self.layerId=lNo
        self.bias=np.random.rand()

    def define_input_weights(self,npl,weights=None):
        if weights != None:
            self.inputconnect=weights[:]
        else:
            for i in range(npl):
                self.inputconnect.append(random.uniform(-2,2))

    def activation_func(self,func,layer=None):
        if(func == 'softmax'):
            numerator=np.exp(self.calculated_value)
            denominator=0
            for obj in layer:
                denominator += np.exp(obj.calculated_value)
            self.activation_value=numerator/denominator
        if(func == 'softmax1'):
            m=-9999
            for obj in layer:
                if m<obj.calculated_value:
                    m=obj.calculated_value
            numerator=np.exp(self.calculated_value-m)
            denominator=0
            for obj in layer:
                denominator += np.exp(obj.calculated_value)
            denominator -= m
            self.activation_value=numerator/denominator
        if(func == 'sigmoid'):
            self.activation_value=1/(1+np.exp(-self.calculated_value))
        if(func == 'relu'):
            self.activation_value=max(0,self.calculated_value)

class neural_network(neuron):

    def __init__(self):
        self.i,self.h1,self.o=[],[],[]
        self.predicted=[]
        self.f1score=0

    def initial_structure_definition(self,p,oc,neurons_in_hlayer):
        self.param=p
        self.oc=oc
        count=0
        count+=1
        for i in range(self.param):
            self.i.append(neuron(count))
        count+=1
        for i in range(neurons_in_hlayer):
            obj=neuron(count)
            obj.define_input_weights(self.param)
            self.h1.append(obj)
        count+=1
        for i in range(self.oc):
            obj=neuron(count)
            obj.define_input_weights(len(self.h1))
            self.o.append(obj)

    def existing_structure_definition(self,p,oc,nuerons_inhlayer,w_h,w_o,b_h,b_o):
        self.param=p
        self.oc=oc
        count=0
        count+=1
        for i in range(self.param):
            self.i.append(neuron(count))
        count+=1
        for i in range(nuerons_inhlayer):
            obj=neuron(count)
            obj.define_input_weights(self.param,w_h[i])
            obj.bias=b_h[i]
            self.h1.append(obj)
        count+=1
        for i in range(self.oc):
            obj=neuron(count)
            obj.define_input_weights(len(self.h1),w_o[i])
            obj.bias=b_o[i]
            self.o.append(obj)

    def forward_pass(self,In,choice=''):
        '''Input Layer'''
        for j in range(len(self.i)):
            self.i[j].activation_value=In[j]
        
        '''First Hidden Layer'''
        for j in range(len(self.h1)):
            act=0
            for k in range(len(self.i)):
                act+= self.i[k].activation_value*self.h1[j].inputconnect[k]
            act += self.h1[j].bias
            self.h1[j].calculated_value = act
        for j in range(len(self.h1)):
            self.h1[j].activation_func('relu')
        

        '''Output Layer'''
        for j in range(len(self.o)):
            act=0
            for k in range(len(self.h1)):
                act+= self.h1[k].activation_value*self.o[j].inputconnect[k]
            act += self.o[j].bias
            self.o[j].calculated_value = act
        for j in range(len(self.o)):
            if choice == 'softmax' :  self.o[j].activation_func('softmax',self.o)
            if choice == 'softmax1' :  self.o[j].activation_func('softmax1',self.o)

    def check_cases(self,output_cases):
        index=-1
        maximum=-9999
        for x in range(len(self.o)):
            if(maximum < self.o[x].activation_value):
                maximum=self.o[x].activation_value
                index=x
        self.predicted.append(output_cases[index])

    def confusion_matrix_implementation(self,Yt):
        self.matrix=confusion_matrix(Yt,self.predicted)
        self.f1score=f1_score(Yt,self.predicted,average='macro')
        pass