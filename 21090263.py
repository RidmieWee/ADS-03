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
from sklearn import cluster as sk_cluster
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


def calculate_silhoutte_score(df):

    print("n    score")
    # loop over number of clusters
    for ncluster in range(2, 10):

        # set up the clusterer with the number of expected clusters
        kmeans = cluster.KMeans(n_clusters=ncluster)

        # Fit the data, results are stored in the kmeans object
        kmeans.fit(df)     # fit done on x,y pairs

        labels = kmeans.labels_

        # extract the estimated cluster centres
        cen = kmeans.cluster_centers_

        # calculate the silhoutte score
        print(ncluster, skmet.silhouette_score(df, labels))

    return


def plot_normalized_cg(df, n, country_names):

    # number of cluster centres
    nc = n

    # define k means
    kmeans = sk_cluster.KMeans(n_clusters=nc)
    kmeans.fit(df)

    # extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_

    # Create a dictionary to store the country names for each cluster
    cluster_country_map = {}

    # assign the country name to the corresponding cluster
    for i, label in enumerate(labels):
        if label not in cluster_country_map:
            cluster_country_map[label] = []
        cluster_country_map[label].append(country_names[i])

    # print the country names for each cluster
    for cluster, countries in cluster_country_map.items():
        print(f"Cluster {cluster}:")
        print(countries)

    # plot the figure
    plt.figure(figsize=(6.0, 6.0))

    # scatter plot with colours selected using the cluster numbers
    plt.scatter(df["CO2"], df["GDP"], c=labels, cmap="tab10")

    # show cluster centres
    xc = cen[:, 0]
    yc = cen[:, 1]

    # plot the scatter plot
    plt.scatter(xc, yc, c="k", marker="d", s=80)

    # add title, labels, and legends
    plt.xlabel("CO2")
    plt.ylabel("GDP")
    plt.title("3 clusters")

    # show the plot
    plt.show()


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

# get the merged data for co2 and gdp for 2019
df_2019_cg = merge_data(df_co2_year, df_gdp_year, "2013")

# explore merged dataframe
print(df_2019_cg.describe())

# rename columns
df_2019_cg = df_2019_cg.rename(columns={"2013_x": "CO2", "2013_y": "GDP"})

# plot the scatter matrix
pd.plotting.scatter_matrix(df_2019_cg, figsize=(12, 12), s=20, alpha=0.8)

# calculate the correlation
print(df_2019_cg.corr())

# copy dataframe to another
df_co2_gdp_2019 = df_2019_cg[["CO2", "GDP"]].copy()

# normalise the dataframe
df_co2_gdp_2019, df_min1, df_max1 = ct.scaler(df_co2_gdp_2019)

# get the merged data for co2 and renewable energy for 2019
df_2019_cr = merge_data(df_co2_year, df_rec_year, "2013")

# explore merged dataframe
print(df_2019_cr.describe())

# rename columns
df_2019_cr = df_2019_cr.rename(columns={"2013_x": "CO2",
                                        "2013_y": "Renewable energy"})

# plot the scatter matrix
pd.plotting.scatter_matrix(df_2019_cr, figsize=(12, 12), s=20, alpha=0.8)

# calculate the correlation
print(df_2019_cr.corr())

# copy dataframe to another
df_co2_rec_2019 = df_2019_cr[["CO2", "Renewable energy"]].copy()

# normalise the dataframe
df_co2_rec_2019, df_min2, df_max2 = ct.scaler(df_co2_rec_2019)

# get the merged data for co2 and renewable energy for 2019
df_2013_ge = merge_data(df_gdp_year, df_energy_year, "2013")

# explore the merged dataframe
df_2013_ge.describe()

# rename columns
df_2013_ge = df_2013_ge.rename(columns={"2013_x": "GDP",
                                        "2013_y": "Energy use"})

# plot the scatter matrix
pd.plotting.scatter_matrix(df_2013_ge, figsize=(12, 12), s=20, alpha=0.8)

# calculate the correlation
print(df_2013_ge.corr())

# copy dataframe to another
df_gdp_en_2019 = df_2013_ge[["GDP", "Energy use"]].copy()

# normalise the dataframe
df_gdp_en_2019, df_min3, df_max3 = ct.scaler(df_gdp_en_2019)

# calculate silhoutte score
print("Score for CO2/GDP:")
calculate_silhoutte_score(df_co2_gdp_2019)
print("Score for CO2/REC:")
calculate_silhoutte_score(df_co2_rec_2019)
print("Score for GDP/EC:")
calculate_silhoutte_score(df_gdp_en_2019)


# get country names
country_names_cg = df_2019_cg["Country Name"].tolist()

# call function to plot normalized cluster
plot_normalized_cg(df_co2_gdp_2019, 2, country_names_cg)












