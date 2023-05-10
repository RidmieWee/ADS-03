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
import scipy.optimize as opt

import cluster_tools as ct
import errors as err


def read_data_dile(filename):

    # read data from csv
    df_data = pd.read_csv(filename, skiprows=4)

    # drop all unnecessary columns
    df_data = df_data.drop(["Indicator Code", "Unnamed: 66"], axis=1)

    # create a dataframe to get years as columns
    df_year = df_data.copy()

    # create a dataframe to get country as columns
    df_country = df_data.copy()

    # drop the years between 1960 to 1990
    df_country = df_data.drop(df_data.iloc[:, 3:33], axis=1)

    # set the country name as index
    df_country = df_country.set_index("Country Name")

    # transpose the dataframe to get countries as columns
    df_country = df_country.transpose()

    # clean the transposed dataframe
    df_country = df_country.dropna(axis=1)

    return df_year, df_country


def merge_data(df1, df2, year):

    # drop rows with NaN's in 1st df
    df_1 = df1[df1[year].notna()]

    # drop rows with NaN's in 2nd df
    df_2 = df2.dropna(subset=[year])

    # extract necessary columns
    df_1 = df_1[["Country Name", "Country Code", year]].copy()
    df_2 = df_2[["Country Name", "Country Code", year]].copy()

    # merge 2 dataframes
    df_merged = pd.merge(df_1, df_2, on="Country Name", how="outer")

    # clean merged dataframe
    df_merged = df_merged.dropna()

    # return merged dataframe
    return df_merged


# read co2 file
df_co2_year, df_co2_country = read_data_dile("CO2.csv")

# read gdp file
df_gdp_year, df_gdp_country = read_data_dile("GDP.csv")

# read renew. energy file
df_rec_year, df_rec_country = read_data_dile("Renewable energy.csv")

# read renew. energy file
df_energy_year, df_energy_country = read_data_dile("Energy use.csv")

# explore the co2 dataframe
print(df_co2_year.describe())

# explore the gdp dataframe
print(df_gdp_year.describe())

# explore the renew. energy dataframe
print(df_rec_year.describe())

# explore the energy consump. dataframe
print(df_energy_year.describe())
