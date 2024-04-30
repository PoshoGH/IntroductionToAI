import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pandas.read_csv('../Datasets/data.csv')

features = ['Car', 'Model', 'Volume', 'Weight', 'C02']
C02 = ['C02']

X = df[features]
y = df[C02]

print(X)
print(y)