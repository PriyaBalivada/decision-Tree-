import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.tree import DecisionTreeClassifier

d=pd.read_csv("/content/Iris (1).csv")
d.head()

dummies=pd.get_dummies(d.iloc[:,-1])
dummies

merge=pd.concat([d,dummies],axis='columns')
merge

final=merge.drop(['Species'],axis='columns')
final.head()

final=final.drop(['Iris-setosa'],axis='columns')
final.head()
