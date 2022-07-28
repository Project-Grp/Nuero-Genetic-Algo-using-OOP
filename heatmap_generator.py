import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def draw_heatmap(net_obj,output_cases):
    cm_df=pd.DataFrame(net_obj.matrix,index=output_cases,columns=output_cases)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_df, cmap='YlGnBu', annot=True)
    plt.title('confusion matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()