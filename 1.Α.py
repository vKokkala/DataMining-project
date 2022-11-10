from tkinter import END
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from pandas.errors import EmptyDataError

sources = ""
demand = ""
sources_exported_file = "sources_united.csv"
demand_exported_file = "demand_united.csv"

able_to_be_negative_sources = ["Time","Batteries","Large Hydro merged","Imports","Other"]

# Basic metrics Section
def show_statistic_metrics(df):
    for column in df.columns:
       if column != "Time":
          print(column) 
          print("mean:"+"\t"+str(df[column].mean()))
          print("deviation:"+"\t"+str(df[column].std()))
          print("median:"+"\t"+str(df[column].median()))
          print("mode:"+"\t"+str(df[column].mode())) 
    
# Analyze dataset Section


# 1.Fill the missing values - interpolation
def fillin_the_missing_values(df):
    # drop_the_duplicates(df)
    df.drop_duplicates(inplace = True)
    #forward interpolation instead of backwards.This way,when all the samples
    #from the start up to a certain point are null,
    #they all shall be filled with the nect following value.
    df.interpolate(method='linear', limit_direction='forward', inplace=True)
     
# unify Csv files Section
def clean_csv(path, exported,type):
    
    files = os.path.join(path, "20*.csv")
    files = glob.glob(files) 
    files_list = []
 
    #put all the files into a list
    for i in range(0, len(files)):
        try:
            pd.read_csv(files[i])
            files_list.append(files[i])
        except EmptyDataError:
            continue
    #unify the list into one
    df = pd.concat(map(pd.read_csv, files_list), ignore_index=True)
    df.to_csv(exported, index=False)  #export to csv
     
    #if we're cleaning the sources, merge the four broken columns and the negative values- wrong values 
    if type == 0: 
     #merge seperated columns
     df['Natural Gas merged'] = df['Natural Gas'].fillna(0) + df['Natural gas'].fillna(0)
     df['Large Hydro merged'] = df['Large Hydro'].fillna(0) + df['Large hydro'].fillna(0)

     #drop function which is used in removing or deleting rows or columns from the CSV files
     df.drop('Natural gas', inplace=True, axis=1)
     df.drop('Natural Gas', inplace=True, axis=1)
     df.drop('Large Hydro', inplace=True, axis=1)
     df.drop('Large hydro', inplace=True, axis=1)  
    
     #Clean the negative values out of specified sources.
     #Fill with none so that the interpolation will fix it later on    
     for column in df.columns:
        if column not in able_to_be_negative_sources:
          for row in df.index:
            if df.loc[row,column] < 0: df.loc[row,column]= None
    
    fillin_the_missing_values(df)
    
    #correct wrong time format
    df['Time'] = pd.to_datetime(df['Time']).dt.strftime("%H:%M")

    df.to_csv(exported, index=False)  # to eniaio csv file
    
    return df

df_sources = clean_csv(sources, sources_exported_file,0) 
df_demand = clean_csv(demand, demand_exported_file,1)

"""
plt.scatter(df_sources.Time,df_sources.Solar)
plt.xlabel("Time")
plt.ylabel("Solar")
plt.show()
"""

show_statistic_metrics(df_sources)
show_statistic_metrics(df_demand)

