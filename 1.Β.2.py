from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from kneed import KneeLocator

df = pd.read_csv('sourcesunited.csv')
dfdemand = pd.read_csv('demandsunited.csv')
per_day_values = list()
temp_list = list()
step = 288
days = 1096
per_day_df = pd.DataFrame()

for column in df.columns:
    for day in range(0, 1096):
        per_day = df.loc[day * step:(day + 1) * step - 1, column].sum()
        per_day_values.append(per_day)

    per_day_df[column] = per_day_values
    temp_list.clear()
    per_day_values.clear()   


def kmeansleitourgiko(numofclusters):
    scaler = MinMaxScaler()

    scaler.fit(df[["Solar"]])
    df["Solar"] = scaler.transform(df[["Solar"]])

    scaler.fit(df[["Batteries"]])
    df["Batteries"] = scaler.transform(df[["Batteries"]])
    
    
    ax = plt.subplots(figsize=(12, 6))
    ax.scatter(per_day_df["Solar"], per_day_df["Batteries"])
    ax.set_xlabel("Solar")
    ax.set_ylabel("Batteries")
    plt.show()
    
    
    kn = KMeans(n_clusters=numofclusters)
    y_predicted = kn.fit_predict(per_day_df[["Solar", "Batteries"]])
    per_day_df['cluster'] = y_predicted
    per_day_df.head()
    kn.cluster_centers_
    
    
    df1 = per_day_df[per_day_df['cluster'] == 0]
    df2 = per_day_df[per_day_df['cluster'] == 1]
    df3 = per_day_df[per_day_df['cluster'] == 2]
    df4 = per_day_df[per_day_df['cluster'] == 3]
    df5 = per_day_df[per_day_df['cluster'] == 4]
    plt.scatter(df1.Solar, df1["Batteries"], color='green')
    plt.scatter(df2.Solar, df2["Batteries"], color='red')
    plt.scatter(df3.Solar, df3["Batteries"], color='black')
    plt.scatter(df4.Solar, df4["Batteries"], color='blue')
    plt.scatter(df5.Solar, df5["Batteries"], color='yellow')
    plt.scatter(kn.cluster_centers_[:, 0], kn.cluster_centers_[:, 1], color='purple', marker='*', label='centroid')
    plt.xlabel("Solar")
    plt.ylabel("Batteries")
    plt.legend()
    plt.show()


sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(per_day_df[['Solar','Batteries']])
    sse.append(km.inertia_)
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.show()
kmeansleitourgiko(1)
