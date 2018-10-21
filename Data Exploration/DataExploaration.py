# imports
import numpy as np    #Load the numpy library for fast array computations
import pandas as pd   #Load the pandas data-analysis library
import matplotlib.pyplot as plt   #Load the pyplot visualization library
from sklearn import datasets
from sklearn.metrics.pairwise import cosine_similarity
from pylab import *
#from mpl_toolkits.mplot3d import Axes3D

# load iris
iris = datasets.load_iris()
X = iris.data 
target = iris.target 
names = iris.target_names     #['setosa' 'versicolor' 'virginica']

#attributes
#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
##########################################
#Cosine Similarity
##########################################

arr = np.random.rand(150,150)
for i in range (0,150):
  for j in range (0,150):
      arr[i][j] = cosine_similarity(X[i].reshape(1,-1), X[j].reshape(1,-1))
#      print(cosine_similarity(X[i].reshape(1,-1), X[j].reshape(1,-1)))
#      print ( arr[i][j])
imshow(arr)

##########################################
## Q1: plot X data for each class 
##########################################

plt.plot(np.transpose(X[:50,0:4 ]))
plt.plot(np.transpose(X[51:100,0:4 ]))
plt.plot(np.transpose(X[101:150,0:4 ]))

##########################################
## Q2: Plot the histogram for each class 
##########################################

plt.hist(X[:50,:4])
plt.hist(X[51:100,:4])
plt.hist(X[101:150,:4])

plt.hist(X[:50,:1])
plt.hist(X[:50,1:2])
plt.hist(X[:50,2:3])
plt.hist(X[:50,3:4])

plt.hist(X[51:100,:1])
plt.hist(X[51:100,1:2])
plt.hist(X[51:100,2:3])
plt.hist(X[51:100,3:4])

plt.hist(X[101:150,:1])
plt.hist(X[101:150,1:2])
plt.hist(X[101:150,2:3])
plt.hist(X[101:150,3:4])

##########################################
## Q3 : plot every 2 attributes together 
##########################################


## sepal length, sepal width

features = iris.data[: , [0,1,2,3]]
features.shape
featuresAll = []
targets = []
for feature in features:
    featuresAll.append(feature[0]) #Sepal length
    targets.append(feature[1]) #sepal width

groups = ('Iris-setosa','Iris-versicolor','Iris-virginica')
colors = ('blue', 'green','red')
data = ((featuresAll[:50], targets[:50]), (featuresAll[50:100], targets[50:100]), 
        (featuresAll[100:150], targets[100:150]))

for item, color, group in zip(data,colors,groups): 
    x, y = item
    plt.scatter(x, y,color=color,alpha=1)
    plt.title('Iris Dataset scatter Plot')
plt.xlabel('sepal length')
plt.ylabel('Sepal width')
plt.show()
 
## sepal length, petal length

featuresAll = []
targets = []
for feature in features:
    featuresAll.append(feature[0]) #Sepal length
    targets.append(feature[2]) #Petal length

groups = ('Iris-setosa','Iris-versicolor','Iris-virginica')
colors = ('blue', 'green','red')
data = ((featuresAll[:50], targets[:50]), (featuresAll[50:100], targets[50:100]), 
        (featuresAll[100:150], targets[100:150]))

for item, color, group in zip(data,colors,groups): 
    x, y = item
    plt.scatter(x, y,color=color,alpha=1)
    plt.title('Iris Dataset scatter Plot')
plt.xlabel('sepal length')
plt.ylabel('Petal length')
plt.show()

## sepal length and petal width

featuresAll = []
targets = []
for feature in features:
    featuresAll.append(feature[0]) #Sepal length
    targets.append(feature[3]) #Petal width

groups = ('Iris-setosa','Iris-versicolor','Iris-virginica')
colors = ('blue', 'green','red')
data = ((featuresAll[:50], targets[:50]), (featuresAll[50:100], targets[50:100]), 
        (featuresAll[100:150], targets[100:150]))

for item, color, group in zip(data,colors,groups): 
    x, y = item
    plt.scatter(x, y,color=color,alpha=1)
    plt.title('Iris Dataset scatter Plot')
plt.xlabel('sepal length')
plt.ylabel('Petal width')
plt.show()

## sepal width and petal length

featuresAll = []
targets = []
for feature in features:
    featuresAll.append(feature[1]) #sepal width
    targets.append(feature[2]) #Petal length

groups = ('Iris-setosa','Iris-versicolor','Iris-virginica')
colors = ('blue', 'green','red')
data = ((featuresAll[:50], targets[:50]), (featuresAll[50:100], targets[50:100]), 
        (featuresAll[100:150], targets[100:150]))

for item, color, group in zip(data,colors,groups): 
    x, y = item
    plt.scatter(x, y,color=color,alpha=1)
    plt.title('Iris Dataset scatter Plot')
plt.xlabel('sepal width')
plt.ylabel('Petal length')
plt.show()

## sepal width and petal width
featuresAll = []
targets = []
for feature in features:
    featuresAll.append(feature[1]) #sepal width
    targets.append(feature[3]) #Petal width

groups = ('Iris-setosa','Iris-versicolor','Iris-virginica')
colors = ('blue', 'green','red')
data = ((featuresAll[:50], targets[:50]), (featuresAll[50:100], targets[50:100]), 
        (featuresAll[100:150], targets[100:150]))

for item, color, group in zip(data,colors,groups): 
    x, y = item
    plt.scatter(x, y,color=color,alpha=1)
    plt.title('Iris Dataset scatter Plot')
plt.xlabel('sepal width')
plt.ylabel('Petal width')
plt.show()


#Finding the relationship between Petal Length and Petal width

featuresAll = []
targets = []
for feature in features:
    featuresAll.append(feature[2]) #Petal length
    targets.append(feature[3]) #Petal width

groups = ('Iris-setosa','Iris-versicolor','Iris-virginica')
colors = ('blue', 'green','red')
data = ((featuresAll[:50], targets[:50]), (featuresAll[50:100], targets[50:100]), 
        (featuresAll[100:150], targets[100:150]))

for item, color, group in zip(data,colors,groups): 
    x0, y0 = item
    plt.scatter(x0, y0,color=color,alpha=1)
    plt.title('Iris Dataset scatter Plot')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.show()

#################################################################################
##Q3 another solution
#################################################################################
# data imports
from pandas import DataFrame
from mpl_toolkits.mplot3d import Axes3D
# plot imports
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
Y = iris.target
# create dataframes for visualisations

iris_data = DataFrame(X, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
iris_target = DataFrame(Y, columns=['Species'])
# at the moment we have 0, 1 and2 for species, so we want to change that to make it clearer

def flower(num):
    if num == 0:
        return 'Setosa'
    elif num == 1:
        return 'Versicolour'
    else:
        return 'Virginica'
 # label flowers# combine dataframes

iris_target['Species'] = iris_target['Species'].apply(flower)   
# combine dataframes

iris = pd.concat([iris_data, iris_target], axis=1)
sns.pairplot(iris, hue='Species', size=2)
#############################################################
## Q4:Use 3D scatter plot to plot every 3 attributes together
#############################################################
#load dataset csv
data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/iris.csv')

global graph #figure
#Function scatter_plot group data by argument name, plot and edit labels
def scatter_plot(x_label,y_label,z_label,clase,c,m,label):
    x = data[ data['Name'] == clase ][x_label] #groupby Name column x_label
    y = data[ data['Name'] == clase ][y_label]
    z = data[ data['Name'] == clase ][z_label]
    # s: size point; alpha: transparent 0, opaque 1; label:legend
    graph.scatter(x,y,z,color=c, edgecolors='k',s=50, alpha=0.9, marker=m,label=label)
    graph.set_xlabel(x_label)
    graph.set_ylabel(y_label)
    graph.set_zlabel(z_label)
    return 

graph = plt.figure().gca(projection='3d')  #new figure
scatter_plot('SepalLength','SepalWidth','PetalLength','Iris-virginica','g','o','Iris-virginica')
scatter_plot('SepalLength','SepalWidth','PetalLength','Iris-versicolor','b','o','Iris-versicolor')
scatter_plot('SepalLength','SepalWidth','PetalLength','Iris-setosa','r','o','Iris-setosa')
plt.legend()
plt.show()

graph = plt.figure().gca(projection='3d')  #new figure
scatter_plot('SepalLength','PetalLength','SepalWidth','Iris-virginica','g','o','Iris-virginica')
scatter_plot('SepalLength','PetalLength','SepalWidth','Iris-versicolor','b','o','Iris-versicolor')
scatter_plot('SepalLength','PetalLength','SepalWidth','Iris-setosa','r','o','Iris-setosa')
plt.legend()
plt.show()

graph = plt.figure().gca(projection='3d')  #new figure
scatter_plot('SepalWidth','SepalLength','PetalLength','Iris-virginica','g','o','Iris-virginica')
scatter_plot('SepalWidth','SepalLength','PetalLength','Iris-versicolor','b','o','Iris-versicolor')
scatter_plot('SepalWidth','SepalLength','PetalLength','Iris-setosa','r','o','Iris-setosa')
plt.legend()
plt.show()

graph = plt.figure().gca(projection='3d')  #new figure
scatter_plot('SepalWidth','PetalLength','SepalLength','Iris-virginica','g','o','Iris-virginica')
scatter_plot('SepalWidth','PetalLength','SepalLength','Iris-versicolor','b','o','Iris-versicolor')
scatter_plot('SepalWidth','PetalLength','SepalLength','Iris-setosa','r','o','Iris-setosa')
plt.legend()
plt.show()

graph = plt.figure().gca(projection='3d')  #new figure
scatter_plot('PetalLength','SepalLength','SepalWidth','Iris-virginica','g','o','Iris-virginica')
scatter_plot('PetalLength','SepalLength','SepalWidth','Iris-versicolor','b','o','Iris-versicolor')
scatter_plot('PetalLength','SepalLength','SepalWidth','Iris-setosa','r','o','Iris-setosa')
plt.legend()
plt.show()

graph = plt.figure().gca(projection='3d')  #new figure
scatter_plot('PetalLength','SepalWidth','SepalLength','Iris-virginica','g','o','Iris-virginica')
scatter_plot('PetalLength','SepalWidth','SepalLength','Iris-versicolor','b','o','Iris-versicolor')
scatter_plot('PetalLength','SepalWidth','SepalLength','Iris-setosa','r','o','Iris-setosa')
plt.legend()
plt.show()







graph = plt.figure().gca(projection='3d')  #new figure
scatter_plot('SepalLength','SepalWidth','PetalWidth','Iris-virginica','g','o','Iris-virginica')
scatter_plot('SepalLength','SepalWidth','PetalWidth','Iris-versicolor','b','o','Iris-versicolor')
scatter_plot('SepalLength','SepalWidth','PetalWidth','Iris-setosa','r','o','Iris-setosa')
plt.legend()
plt.show()
graph = plt.figure().gca(projection='3d')  #new figure
scatter_plot('SepalLength','PetalWidth','SepalWidth','Iris-virginica','g','o','Iris-virginica')
scatter_plot('SepalLength','PetalWidth','SepalWidth','Iris-versicolor','b','o','Iris-versicolor')
scatter_plot('SepalLength','PetalWidth','SepalWidth','Iris-setosa','r','o','Iris-setosa')
plt.legend()
plt.show()

graph = plt.figure().gca(projection='3d')  #new figure
scatter_plot('SepalWidth','SepalLength','PetalWidth','Iris-virginica','g','o','Iris-virginica')
scatter_plot('SepalWidth','SepalLength','PetalWidth','Iris-versicolor','b','o','Iris-versicolor')
scatter_plot('SepalWidth','SepalLength','PetalWidth','Iris-setosa','r','o','Iris-setosa')
plt.legend()
plt.show()

graph = plt.figure().gca(projection='3d')  #new figure
scatter_plot('SepalWidth','PetalWidth','SepalLength','Iris-virginica','g','o','Iris-virginica')
scatter_plot('SepalWidth','PetalWidth','SepalLength','Iris-versicolor','b','o','Iris-versicolor')
scatter_plot('SepalWidth','PetalWidth','SepalLength','Iris-setosa','r','o','Iris-setosa')
plt.legend()
plt.show()

graph = plt.figure().gca(projection='3d')  #new figure
scatter_plot('PetalWidth','SepalLength','SepalWidth','Iris-virginica','g','o','Iris-virginica')
scatter_plot('PetalWidth','SepalLength','SepalWidth','Iris-versicolor','b','o','Iris-versicolor')
scatter_plot('PetalWidth','SepalLength','SepalWidth','Iris-setosa','r','o','Iris-setosa')
plt.legend()
plt.show()

graph = plt.figure().gca(projection='3d')  #new figure
scatter_plot('PetalWidth','SepalWidth','SepalLength','Iris-virginica','g','o','Iris-virginica')
scatter_plot('PetalWidth','SepalWidth','SepalLength','Iris-versicolor','b','o','Iris-versicolor')
scatter_plot('PetalWidth','SepalWidth','SepalLength','Iris-setosa','r','o','Iris-setosa')
plt.legend()
plt.show()






graph = plt.figure().gca(projection='3d')  #new figure
scatter_plot('PetalLength','SepalWidth','PetalWidth','Iris-virginica','g','o','Iris-virginica')
scatter_plot('PetalLength','SepalWidth','PetalWidth','Iris-versicolor','b','o','Iris-versicolor')
scatter_plot('PetalLength','SepalWidth','PetalWidth','Iris-setosa','r','o','Iris-setosa')
plt.legend()
plt.show()
graph = plt.figure().gca(projection='3d')  #new figure
scatter_plot('PetalLength','PetalWidth','SepalWidth','Iris-virginica','g','o','Iris-virginica')
scatter_plot('PetalLength','PetalWidth','SepalWidth','Iris-versicolor','b','o','Iris-versicolor')
scatter_plot('PetalLength','PetalWidth','SepalWidth','Iris-setosa','r','o','Iris-setosa')
plt.legend()
plt.show()
graph = plt.figure().gca(projection='3d')  #new figure
scatter_plot('SepalWidth','PetalLength','PetalWidth','Iris-virginica','g','o','Iris-virginica')
scatter_plot('SepalWidth','PetalLength','PetalWidth','Iris-versicolor','b','o','Iris-versicolor')
scatter_plot('SepalWidth','PetalLength','PetalWidth','Iris-setosa','r','o','Iris-setosa')
plt.legend()
plt.show()
graph = plt.figure().gca(projection='3d')  #new figure
scatter_plot('SepalWidth','PetalWidth','PetalLength','Iris-virginica','g','o','Iris-virginica')
scatter_plot('SepalWidth','PetalWidth','PetalLength','Iris-versicolor','b','o','Iris-versicolor')
scatter_plot('SepalWidth','PetalWidth','PetalLength','Iris-setosa','r','o','Iris-setosa')
plt.legend()
plt.show()
graph = plt.figure().gca(projection='3d')  #new figure
scatter_plot('PetalWidth','PetalLength','SepalWidth','Iris-virginica','g','o','Iris-virginica')
scatter_plot('PetalWidth','PetalLength','SepalWidth','Iris-versicolor','b','o','Iris-versicolor')
scatter_plot('PetalWidth','PetalLength','SepalWidth','Iris-setosa','r','o','Iris-setosa')
plt.legend()
plt.show()
graph = plt.figure().gca(projection='3d')  #new figure
scatter_plot('PetalWidth','SepalWidth','PetalLength','Iris-virginica','g','o','Iris-virginica')
scatter_plot('PetalWidth','SepalWidth','PetalLength','Iris-versicolor','b','o','Iris-versicolor')
scatter_plot('PetalWidth','SepalWidth','PetalLength','Iris-setosa','r','o','Iris-setosa')
plt.legend()
plt.show()






graph = plt.figure().gca(projection='3d')  #new figure
scatter_plot('PetalLength','SepalLength','PetalWidth','Iris-virginica','g','o','Iris-virginica')
scatter_plot('PetalLength','SepalLength','PetalWidth','Iris-versicolor','b','o','Iris-versicolor')
scatter_plot('PetalLength','SepalLength','PetalWidth','Iris-setosa','r','o','Iris-setosa')
plt.legend()
plt.show()
graph = plt.figure().gca(projection='3d')  #new figure
scatter_plot('PetalLength','PetalWidth','SepalLength','Iris-virginica','g','o','Iris-virginica')
scatter_plot('PetalLength','PetalWidth','SepalLength','Iris-versicolor','b','o','Iris-versicolor')
scatter_plot('PetalLength','PetalWidth','SepalLength','Iris-setosa','r','o','Iris-setosa')
plt.legend()
plt.show()
graph = plt.figure().gca(projection='3d')  #new figure
scatter_plot('SepalLength','PetalLength','PetalWidth','Iris-virginica','g','o','Iris-virginica')
scatter_plot('SepalLength','PetalLength','PetalWidth','Iris-versicolor','b','o','Iris-versicolor')
scatter_plot('SepalLength','PetalLength','PetalWidth','Iris-setosa','r','o','Iris-setosa')
plt.legend()
plt.show()
graph = plt.figure().gca(projection='3d')  #new figure
scatter_plot('SepalLength','PetalWidth','PetalLength','Iris-virginica','g','o','Iris-virginica')
scatter_plot('SepalLength','PetalWidth','PetalLength','Iris-versicolor','b','o','Iris-versicolor')
scatter_plot('SepalLength','PetalWidth','PetalLength','Iris-setosa','r','o','Iris-setosa')
plt.legend()
plt.show()
graph = plt.figure().gca(projection='3d')  #new figure
scatter_plot('PetalWidth','PetalLength','SepalLength','Iris-virginica','g','o','Iris-virginica')
scatter_plot('PetalWidth','PetalLength','SepalLength','Iris-versicolor','b','o','Iris-versicolor')
scatter_plot('PetalWidth','PetalLength','SepalLength','Iris-setosa','r','o','Iris-setosa')
plt.legend()
plt.show()
graph = plt.figure().gca(projection='3d')  #new figure
scatter_plot('PetalWidth','SepalLength','PetalLength','Iris-virginica','g','o','Iris-virginica')
scatter_plot('PetalWidth','SepalLength','PetalLength','Iris-versicolor','b','o','Iris-versicolor')
scatter_plot('PetalWidth','SepalLength','PetalLength','Iris-setosa','r','o','Iris-setosa')
plt.legend()
plt.show()

