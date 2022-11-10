# import packDemands
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df1 = pd.read_csv('sources_united.csv')
df2 = pd.read_csv('demand_united.csv')
data = {
  "Index": [],
  "Demand": []
}

df = pd.DataFrame(data)
df1 = df.loc[:, df.columns != "Time"].fillna(0)
df['Index'] = df2.index
df["Demand"] = df2["Current demand"].fillna(0)

ss = StandardScaler()
df = pd.DataFrame(ss.fit_transform(df), columns=['Index', 'Demand'])


# K-means clustering
km = KMeans(n_clusters=4)
model = km.fit(df)

# Cluster visualization
colors=["red","blue","green","orange"]

# figure setting
plt.figure(figsize=(8,6))
for i in range(np.max(model.labels_)+1):
    plt.scatter(df[model.labels_==i].Index, df[model.labels_==i].Demand, label=i, c=colors[i], alpha=0.5, s=40)
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], label='Centers', c="black", s=100)

def distance_from_center(Index, Demand, label):
    center_Index =  model.cluster_centers_[label,0]
    center_Demand =  model.cluster_centers_[label,1]
    distance = np.sqrt((Index - center_Index) ** 2 + (Demand - center_Demand) ** 2)
    return np.round(distance, 3)
df['label'] = model.labels_
df['distance'] = distance_from_center(df.Index, df.Demand, df.label)


outliers_idx = list(df.sort_values('distance', ascending=False).head(10).index)
outliers = df[df.index.isin(outliers_idx)]
outliers_idx = list(df.sort_values('distance', ascending=False).head(10).index)
outliers = df[df.index.isin(outliers_idx)]
print(outliers)

plt.figure(figsize=(8,6))
colors=["red","blue","green","orange"]
for i in range(np.max(model.labels_)+1):
    plt.scatter(df[model.labels_==i].Index, df[model.labels_==i].Demand, label=i, c=colors[i], alpha=0.5, s=40)
plt.scatter(outliers.Index, outliers.Demand, c='aqua', s=100)
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], label='Centers', c="black", s=100)
plt.title("K-Means Clustering of df Data",size=20)
plt.xlabel(" Index")
plt.ylabel("Demand")
plt.title('Scatter plot of Index vs. Demand')
plt.legend()
plt.show()