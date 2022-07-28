import numpy as np
import random
from network_structure import neural_network

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

def cross_over(weight_population1,bias_population1,weight_population2,bias_population2,Better_parent,Xt,child_list):
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

def mutation(weight_population1,bias_population1,weight_population2,bias_population2,child):
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

def produce_child(genecombo_weight,genecombo_bias,neurons_in_hlayer,parameter_count,output_case_count):
    wH,wO,bH,bO=[],[],[],[]
    wH=genecombo_weight[0:parameter_count*10]
    wO=genecombo_weight[parameter_count*10:]
    bH=genecombo_bias[0:10]
    bO=genecombo_bias[10:]
    wH=[wH[x:x+parameter_count] for x in range(0,len(wH),parameter_count)]
    wO=[wO[x:x+neurons_in_hlayer] for x in range(0,len(wO),neurons_in_hlayer)]
    obj=neural_network()
    obj.existing_structure_definition(parameter_count,output_case_count,neurons_in_hlayer,wH,wO,bH,bO)
    return obj