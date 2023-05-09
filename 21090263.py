# -*- coding: utf-8 -*-
"""
Created on Sun May  6 13:31:37 2023

@author: Ridmi Weerakotuwa
"""

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import cluster_tools as ct

def read_data_dile(filename):

    # read data from csv
    df_data = pd.read_csv(filename, skiprows=4)

    # drop all unnecessary columns
    df_data = df_data.drop(["Indicator Code", "Unnamed: 66"], axis=1)

    # drop the years between 1960 to 1990
    df_data = df_data.drop(df_data.iloc[:, 3:33], axis=1)

    # create a dataframe to get years as columns
    df_year = df_data.copy()

    # set the country name as index
    df_country = df_data.set_index("Country Name")

    # transpose the dataframe to get countries as columns
    df_country = df_country.transpose()

    # clean the transposed dataframe
    df_country = df_country.dropna(axis=1)

    return df_year, df_country
