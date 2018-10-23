# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 21:38:28 2018

@author: Dahlia
"""

import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv("segmentation.data")
test = pd.read_csv("segmentation.test")
# adding the 2 files together
dataset = pd.concat([data, test])

# data dimensions
# number of attributes is the number of columns
print("data dimensions:", data.shape)
print("test dimensions:", test.shape)
print("merged data dimensions:", dataset.shape)

display(data.head())
display(test.head())
display(dataset.head())


#sort by class and reset index
df=pd.DataFrame(dataset)
dataset=df.sort_index()

# gather information about the data (the info used to in boxplots)
display(dataset.describe())
display (dataset.info())


# plotting
#boxplot
dataset.boxplot(rot=90, figsize=(20,16))

# linear plot
for i in range (7):
    subset=dataset[i*330:i*330+330]
    plt.plot(np.transpose(subset))
    plt.xticks(rotation='vertical')
    plt.suptitle(set(subset.index[subset['REGION-PIXEL-COUNT'] == 9].tolist()))
    plt.show()

# show class
# class here is the index of the dataset
# add class attribute and reset the index
dataset.index.name = "class"
dataset = dataset.reset_index()
# the unique values of class label
dataset ['class'].unique()

########################################################################
#Part I : Data Exploration
########################################################################
#Pearson correlation

pearson_correlation = dataset.corr()
display(pearson_correlation)

# visualization of pearson correlation
plt.figure(figsize = (12,8))
plt.imshow(pearson_correlation)
plt.colorbar()

#########################################################################
# Covariance
covariance = dataset.cov()
display(covariance)

# visualization of covariance matrix
plt.figure(figsize = (12,8))
plt.imshow(covariance)
plt.colorbar()
##########################################################################
# Histograms
classes = dataset["class"].unique()

for class_name in classes:
    plt.figure(figsize=(40,30))
    class_data = dataset[dataset["class"] == class_name].drop("class", axis=1)
    class_data.plot.hist(alpha=0.5, bins=40)
    plt.title(class_name)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.show()
    
# visualizations for bins = 5, 10, 12
for bins in [5,10,12]:
    fig, axes = plt.subplots(nrows=4, ncols=2)
    fig.set_size_inches((15,15))
    fig.suptitle("case:  "+ str(bins) +" bins")
    i,j = 0,0
    for class_name in classes:
        axes[i,j].set_title(class_name)
        class_data = dataset[dataset["class"] == class_name].drop("class", axis=1)
        if(i == 0 and j == 1):
            class_data.plot.hist(alpha=0.5, bins=bins, ax=axes[i,j], legend=True)
        else:
            class_data.plot.hist(alpha=0.5, bins=bins, ax=axes[i,j], legend=False)
        if j%2 == 1:
            j=0
            i+=1
        else:
            j+=1
    axes[0,1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
##############################################################################################
# Part II : Preprocessing
##############################################################################################    

# min-max    
def min_max(data):
    scaled = preprocessing.minmax_scale(data) # numpy array 
    return pd.DataFrame(scaled, columns=data.columns)

mm_scaled_data = min_max(dataset.drop("class", axis=1))
mm_scaled_data.head()
mm_scaled_data.boxplot(rot=90, figsize=(10,8))
mn_matrix=mm_scaled_data.describe()

 # min-max other method
 
a = np.array(dataset)
f_arr = a[:721,1:20]
scaler = MinMaxScaler()
scaler.fit(f_arr)
min_arr = scaler.transform(f_arr)
fig1, ax1 = plt.subplots()
ax1.set_title('min-max box Plot')
ax1.boxplot(min_arr)

# Z score

def Z_score_normalizer(data):
    scaler = preprocessing.StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns)

zs_scaled_data = Z_score_normalizer(dataset.drop("class", axis=1))
zs_scaled_data.head()
zs_scaled_data.boxplot(rot=90, figsize=(10,8))
zs_scaled_data.describe()


# Z score other method

x_np = np.asarray(a[:721,1:20])
z_scores_np = (x_np - x_np.mean()) / x_np.std()
print (z_scores_np)
fig1, ax1 = plt.subplots()
ax1.set_title('z box Plot')
ax1.boxplot(z_scores_np)

# Dimensionality reduction
#####################################################################
# feature extraction
#####################################################################
components_num = [1, 2, 4, 6, 8, 10, 13, 16, 19]
ls = list()
for n in components_num:
    pca = PCA(n_components=n)
    pca.fit(zs_scaled_data)
    #variance captured by each component
    var_ratios = pca.explained_variance_ratio_
    print(var_ratios)
    ls.append(sum(var_ratios))

var_sum_df = pd.DataFrame(ls, columns=["captured variance sum"])
var_sum_df["component number"] = components_num
display(var_sum_df)

# number of components chosen is 14
pca = PCA(n_components=14)
pca.fit(zs_scaled_data)
pca_df = pd.DataFrame(pca.transform(zs_scaled_data))
pca_df.head()
# visualization
plt.figure(figsize = (12,8))
plt.imshow(pca_df.corr())
plt.colorbar()
########################################################################

#Feature selection

kbest = [3, 7, 10, 14, 16, 19]
plt.figure(figsize = (12,20))
i = 1
for k in kbest:
    kbest_data_arr = SelectKBest(chi2, k).fit_transform(mm_scaled_data, dataset["class"])
    kbest_data = pd.DataFrame(kbest_data_arr)
    print (kbest_data)
    # visualization
    plt.subplot(3,2,i)
    i+=1
    plt.imshow(kbest_data.corr())
    plt.colorbar()
    plt.title("k = "+str(k))
