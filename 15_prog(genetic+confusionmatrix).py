import random
import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

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
        for i in range(neurons_in_hlayer):
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

    def forward_pass(self,In):
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
            self.o[j].activation_func('softmax',self.o)

    def check_cases(self):
        index=-1
        maximum=-9999
        for x in range(len(self.o)):
            if(maximum < self.o[x].activation_value):
                maximum=self.o[x].activation_value
                index=x
        self.predicted.append(output_cases[index])

    def confusion_matrix_implementation(self):
        self.matrix=confusion_matrix(Yt,self.predicted)
        self.f1score=f1_score(Yt,self.predicted,average='macro')
        pass


''' Genetic Portion'''

def initial_population(obj1,obj2):
    weight_population1,bias_population1,weight_population2,bias_population2=[],[],[],[]
    for object in obj1.h1:
        weight_population1=weight_population1 + object.inputconnect[:]
    for object in obj1.o:
        weight_population1=weight_population1 + object.inputconnect[:]

    for object in obj2.h1:
        weight_population2=weight_population2 + object.inputconnect[:]
    for object in obj2.o:
        weight_population2=weight_population2 + object.inputconnect[:]

    for object in obj1.h1:
        bias_population1.append(object.bias)
    for object in obj1.o:
        bias_population1.append(object.bias)

    for object in obj2.h1:
        bias_population2.append(object.bias)
    for object in obj2.o:
        bias_population2.append(object.bias)
    return weight_population1,bias_population1,weight_population2,bias_population2

def cross_over(weight_population1,bias_population1,weight_population2,bias_population2,Better_parent):
    selection_list=[Better_parent]
    w_1p,b_1p=0,0
    w_1p=random.randint(0,(len(weight_population1)//2)-1)
    b_1p=random.randint(0,(len(bias_population1)//2)-1)
    genecombo_weight,genecombo_bias=[],[]
    for i in range(4):
        genecombo_weight.append([0 for x in range(len(weight_population1))])
        genecombo_bias.append([0 for x in range(len(bias_population1))])

    for i in range(len(weight_population1)):
        if(i < w_1p):
            genecombo_weight[0][i]=weight_population1[i]
            genecombo_weight[1][i]=weight_population2[i]
        else:
            genecombo_weight[0][i]=weight_population2[i]
            genecombo_weight[1][i]=weight_population1[i]

    k=0
    for i in range(w_1p,len(weight_population1)):
        genecombo_weight[2][k]=weight_population2[i]
        genecombo_weight[3][k]=weight_population1[i]
        k+=1
    for i in range(0,w_1p):
        genecombo_weight[2][k]=weight_population1[i]
        genecombo_weight[3][k]=weight_population2[i]
        k+=1
    
    for i in range(len(bias_population1)):
        if(i<b_1p):
            genecombo_bias[0][i]=bias_population1[i]
            genecombo_bias[1][i]=bias_population2[i]
        else:
            genecombo_bias[0][i]=bias_population2[i]
            genecombo_bias[1][i]=bias_population1[i]

    k=0
    for i in range(b_1p,len(bias_population1)):
        genecombo_bias[2][k]=bias_population2[i]
        genecombo_bias[3][k]=bias_population1[i]
        k+=1
    for i in range(0,b_1p):
        genecombo_bias[2][k]=bias_population1[i]
        genecombo_bias[3][k]=bias_population2[i]
        k+=1
    
    for i in range(len(genecombo_weight)):
        child=produce_child(genecombo_weight[i],genecombo_bias[i])
        mutated_child=mutation(child)
        for c in range(len(Xt)):
            case=Xt[c]
            child.forward_pass(case)
            mutated_child.forward_pass(case)
            child.check_cases()
            mutated_child.check_cases()
        child.confusion_matrix_implementation()
        mutated_child.confusion_matrix_implementation()
        selection_list.append(child)
        selection_list.append(mutated_child)

    flag,i=True,len(selection_list)-1
    while(flag == True):
        flag=False
        for j in range(0,i):
            if selection_list[j].f1score < selection_list[j+1].f1score:
                temp=selection_list[j]
                selection_list[j]=selection_list[j+1]
                selection_list[j+1]=temp
                flag=True
        i -= 1
    child_list.append(selection_list[0])
    child_list.append(selection_list[1])

def mutation(child):
    gen_rand_index_count_weight=random.randint(0,len(weight_population1)-1)
    gen_rand_index_count_bias=random.randint(0,len(bias_population1)-1)
    random_weight_index,random_bias_index=[random.randint(0,len(weight_population1)-1) for x in range(gen_rand_index_count_weight)],[random.randint(0,len(bias_population1)-1) for x in range(gen_rand_index_count_bias)]
    random_weight_index,random_bias_index=list(set(random_weight_index)),list(set(random_bias_index))
    weight_list,bias_list=[],[]
    for object in child.h1:
        weight_list=weight_list + object.inputconnect[:]
    for object in child.o:
        weight_list=weight_list + object.inputconnect[:]
    for object in child.h1:
        bias_list.append(object.bias)
    for object in child.o:
        bias_list.append(object.bias)
    contrast_swapping_weight=[weight_list[x] for x in random_weight_index]
    contrast_swapping_bias=[bias_list[x] for x in random_bias_index]
    random.shuffle(contrast_swapping_weight)
    random.shuffle(contrast_swapping_bias)
    k=0
    for i in range(len(weight_list)):
        if(i in random_weight_index):
            weight_list[i]=contrast_swapping_weight[k]
            k+=1
    k=0
    for i in range(len(bias_list)):
        if(i in random_bias_index):
            bias_list[i]=contrast_swapping_bias[k]
            k+=1
    mutant=produce_child(weight_list,bias_list)
    return mutant

''' Graphical Representation'''
def produce_child(genecombo_weight,genecombo_bias):
    wH,wO,bH,bO=[],[],[],[]
    wH=genecombo_weight[0:L[0]*neurons_in_hlayer]
    wO=genecombo_weight[L[0]*neurons_in_hlayer:]
    bH=genecombo_bias[0:neurons_in_hlayer]
    bO=genecombo_bias[neurons_in_hlayer:]
    wH=[wH[x:x+L[0]] for x in range(0,len(wH),L[0])]
    wO=[wO[x:x+neurons_in_hlayer] for x in range(0,len(wO),neurons_in_hlayer)]
    obj=neural_network()
    obj.existing_structure_definition(L[0],L[1],neurons_in_hlayer,wH,wO,bH,bO)
    return obj

'''End of genetic portion'''


def retrieve_data():
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
'''****************************'''


''' Graphical Representation'''
def draw_heatmap():
    cm_df=pd.DataFrame(best_child_list[0].matrix,index=output_cases,columns=output_cases)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_df, cmap='YlGnBu', annot=True)
    plt.title('confusion matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()

def draw_graph():
    plt.plot(x_axis,y_axis,color='red', linewidth = 1, marker='o' , markerfacecolor='blue', markersize=7)
    plt.xlabel('Iteration Number')
    plt.ylabel('Accuracy')
    plt.title('Iteration no. vs Accuracy')
    plt.show()
    pass

def create_weightcsv():
    weight_l,bias_l=[],[]
    for obj in best_child_list[0].h1:
        weight_l.append(obj.inputconnect)
        bias_l.append(obj.bias)
    for obj in best_child_list[0].o:
        weight_l.append(obj.inputconnect)
        bias_l.append(obj.bias)
    tag=[]
    for x in range(neurons_in_hlayer):
        tag.append('h')
    for x in range(L[1]):
        tag.append('w')
    d = {'layer':tag,'bias':bias_l,'weights':weight_l}
    df=pd.DataFrame(d)
    df.to_csv('seedsfile2.csv',index=False)

'''****************************'''

neurons_in_hlayer=10
x_axis,y_axis=[],[]
file_name=input('Enter file name for reading data: ')
L=retrieve_data()
Xt=L[2]
Yt=L[3]
output_cases=L[4]
optimum_network=None
best_child_list=[]
desired_accuracy=float(input('Enter desired accuracy: '))
desired_accuracy /= 100
iteration=int(input('Enter the number of iterations you want to go on for: '))
counter=1
while counter<=iteration:
    x_axis.append(counter)
    print(f'Iteration Number: {counter}:\n\n\n')
    parent_list,child_list=[],[]
    print('  Defining parent networks. Status: ',end=" ")
    for i in range(10):
        obj=neural_network()
        obj.initial_structure_definition(L[0],L[1],neurons_in_hlayer)
        parent_list.append(obj)
    print('Successful!!!!')

    print('  Performing forward on parents and calculating error...Status: ',end=" ")
    for obj in parent_list:
        for c in range(len(Xt)):
            case=Xt[c]
            obj.forward_pass(case)
            obj.check_cases()
        obj.confusion_matrix_implementation()
    print('Successful!!!')

    flag,i=True,len(parent_list)-1
    while(flag == True):
        flag=False
        for j in range(0,i):
            if(parent_list[j].f1score < parent_list[j+1].f1score):
                temp_obj=parent_list[j]
                parent_list[j]=parent_list[j+1]
                parent_list[j+1]=temp_obj
                flag=True
        i -= 1

    generation,val_checker,previous_f1score=0,0,parent_list[0].f1score,
    print('\n\n  Generation table:')
    print(f'  Generation  min_error_caught')
    while parent_list[0].f1score <= desired_accuracy:
        child_list.clear()
        for i in range(0,len(parent_list)-1,2):
            if parent_list[i].f1score > parent_list[i+1].f1score:
                better_parent=parent_list[i]
            else:
                better_parent=parent_list[i+1]
            weight_population1,bias_population1,weight_population2,bias_population2=initial_population(parent_list[i],parent_list[i+1])
            cross_over(weight_population1,bias_population1,weight_population2,bias_population2,better_parent)

        flag,i=True,len(child_list)-1
        while(flag == True):
            flag=False
            for j in range(0,i):
                if(child_list[j].f1score < child_list[j+1].f1score):
                    temp_obj=child_list[j]
                    child_list[j]=child_list[j+1]
                    child_list[j+1]=temp_obj
                    flag=True
            i -= 1
        
        generation += 1
        print(f'  {generation}          {child_list[0].f1score}         {val_checker}')
        parent_list.clear()
        for obj in child_list:
            parent_list.append(obj)
        if(parent_list[0].f1score == previous_f1score):
            val_checker+=1
        else:
            val_checker=1
            previous_f1score=parent_list[0].f1score
        if(val_checker > 1000):
            break

    best_child_list.append(parent_list[0])
    counter+=1
    y_axis.append(parent_list[0].f1score)
    if(counter > iteration):
        # y_axis.clear()
        # for child in best_child_list:
        #     y_axis.append(child.f1score * 100)
        flag,i=True,len(best_child_list)-1
        while(flag == True):
            flag=False
            for j in range(0,i):
                if(best_child_list[j].f1score < best_child_list[j+1].f1score):
                    temp_obj=best_child_list[j]
                    best_child_list[j]=best_child_list[j+1]
                    best_child_list[j+1]=temp_obj
                    flag=True
            i -= 1
        if(best_child_list[0].f1score < desired_accuracy):    
            choice=input(f'\n{best_child_list[0].f1score * 100}% accuracy was acheived. Do you want to go for another iteration?(y/n): ')
            if(choice == 'y'):
                iteration += 1

print(f'Highest accuracy achieved was: {best_child_list[0].f1score * 100}%')

draw_heatmap()
create_weightcsv()
draw_graph()