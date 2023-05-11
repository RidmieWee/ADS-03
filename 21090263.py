# -*- coding: utf-8 -*-
"""
Created on Sun May  6 13:31:37 2023

@author: Ridmi Weerakotuwa
"""


# =============================================================================
# Imports
# =============================================================================

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

# =============================================================================
# Function definitions
# =============================================================================

def read_data_dile(filename):

    """ This functions get file nme as parametr and return dataframes """

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
    """ This function get 2dataframes and according to the given year
    merged into one dataframe """

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
    """ Calculates the silhoutte forgiven df"""

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


# Calculate cluster density
def calculate_cluster_density(labels):
    """ calculate cluster density"""

    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_density = counts / np.sum(counts)
    return cluster_density


# Calculate cluster mean
def calculate_cluster_mean(labels, data):
    """ calculate cluster mean"""

    unique_labels = np.unique(labels)
    cluster_means = []
    for label in unique_labels:
        cluster_data = data[labels == label]
        cluster_mean = np.mean(cluster_data, axis=0)
        cluster_means.append(cluster_mean)
    return np.array(cluster_means)


# Calculate cluster median
def calculate_cluster_median(labels, data):
    """ calculate cluster median"""

    unique_labels = np.unique(labels)
    cluster_medians = []
    for label in unique_labels:
        cluster_data = data[labels == label]
        cluster_median = np.median(cluster_data, axis=0)
        cluster_medians.append(cluster_median)
    return np.array(cluster_medians)


# Calculate cluster standard deviation
def calculate_cluster_std(labels, data):
    """ calculate cluster std """

    unique_labels = np.unique(labels)
    cluster_stds = []
    for label in unique_labels:
        cluster_data = data[labels == label]
        cluster_std = np.std(cluster_data, axis=0)
        cluster_stds.append(cluster_std)
    return np.array(cluster_stds)


# Calculate cluster metrics
def calculate_cluster_metrics(labels, data):
    """ calculate cluster metrics """

    cluster_density = calculate_cluster_density(labels)
    cluster_mean = calculate_cluster_mean(labels, data)
    cluster_median = calculate_cluster_median(labels, data)
    cluster_std = calculate_cluster_std(labels, data)
    return cluster_density, cluster_mean, cluster_median, cluster_std


def plot_normalized_cg(df, n, country_names):
    """This function det dataframe, number of clusters and
    country names and return normalized cluster"""

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
    for cluster1, countries in cluster_country_map.items():
        print(f"Cluster {cluster1}:")
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
    plt.title("CO2 Emissions vs. GDP")

    # show the plot
    plt.show()


def plot_original_scale_cg(dfn, dfo, df_min, df_max, n):
    """This function returns original scale cluster"""

    # number of cluster centres
    nc = n

    # fit normalized data to kmeans algorithm
    kmeans = cluster.KMeans(n_clusters=nc)
    kmeans.fit(dfn)

    # extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_

    # Calculate cluster statistics
    for i in range(nc):
        cluster_data = dfo.loc[labels == i, ["CO2", "GDP"]]
        cluster_density = len(cluster_data)
        cluster_mean = cluster_data.mean()
        cluster_median = cluster_data.median()
        cluster_std = cluster_data.std()

        print(f"2019 Cluster {i+1}:")
        print("Density:", cluster_density)
        print("Mean:", cluster_mean)
        print("Median:", cluster_median)
        print("Standard Deviation:", cluster_std)
        print()

    # plot the figure
    plt.figure(figsize=(6.0, 6.0))

    # Create separate dataframes for countries
    df_canada = dfo[dfo["Country Name"] == "Canada"]
    df_china = dfo[dfo["Country Name"] == "China"]
    df_australia = dfo[dfo["Country Name"] == "Australia"]
    df_brazil = dfo[dfo["Country Name"] == "Brazil"]
    df_germany = dfo[dfo["Country Name"] == "Germany"]
    df_japan = dfo[dfo["Country Name"] == "Japan"]

    # plot the scatter plot
    scatter = plt.scatter(dfo["CO2"],
                          dfo["GDP"],
                          c=labels,
                          cmap="Set3")
    plt.scatter(df_canada["CO2"],
                df_canada["GDP"],
                color='#FFD142',
                label='Canada')
    plt.scatter(df_china["CO2"],
                df_china["GDP"],
                color='#0FEDCF',
                label='China')
    plt.scatter(df_australia["CO2"],
                df_australia["GDP"],
                color='#FFD142',
                label='Australia')
    plt.scatter(df_brazil["CO2"],
                df_brazil["GDP"],
                color='#0FEDCF',
                label='Brazil')
    plt.scatter(df_germany["CO2"],
                df_germany["GDP"],
                color='#FFD142',
                label='Germany')
    plt.scatter(df_japan["CO2"],
                df_japan["GDP"],
                color='#FFD142',
                label='Japan')

    # rescale and show cluster centres
    scen = ct.backscale(cen, df_min, df_max)
    xc = scen[:, 0]
    yc = scen[:, 1]

    # plot scatter plot
    plt.scatter(xc, yc, c="k", marker="d", s=80)

    # Add country names near the data points
    offset = 0.1  # offset the text placement for better visibility
    for x, y, country in zip(dfo["CO2"],
                             dfo["GDP"],
                             dfo["Country Name"]):
        if country in ["Canada", "China", "Australia",
                       "Brazil", "Germany", "Japan"]:
            plt.text(x + offset, y + offset,
                     country, fontsize=11, color="black")

    # Create a legend
    legend_labels = ['Cluster {}'.format(i+1) for i in range(nc)]
    plt.legend(handles=scatter.legend_elements()[0],
               labels=legend_labels, loc="upper right")

    # Rotate y-axis label
    plt.yticks(rotation=45)

    # add legend and title
    plt.xlabel("CO2 (MT per capita)", color="black", fontweight='bold')
    plt.ylabel("GDP per capita (current US$)",
               color="black",
               fontweight='bold')
    plt.title("CO2 Emissions vs. GDP - 2019",
              color="black",
              fontweight='bold', y=1.02)
    plt.show()


def plot_original_scale_cg_1990(dfn, dfo, df_min, df_max, n):
    """This function returns original scale cluster for 1990"""

    # number of cluster centres
    nc = n

    # fit normalized data to kmeans algorithm
    kmeans = cluster.KMeans(n_clusters=nc)
    kmeans.fit(dfn)

    # extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_

    # plot the figure
    plt.figure(figsize=(6.0, 6.0))

    # Create separate dataframes for countries
    df_canada = dfo[dfo["Country Name"] == "Canada"]
    df_china = dfo[dfo["Country Name"] == "China"]
    df_australia = dfo[dfo["Country Name"] == "Australia"]
    df_brazil = dfo[dfo["Country Name"] == "Brazil"]
    df_germany = dfo[dfo["Country Name"] == "Germany"]
    df_japan = dfo[dfo["Country Name"] == "Japan"]

    # plot the scatter plot
    scatter = plt.scatter(dfo["CO2"], dfo["GDP"], c=labels, cmap="Set3")
    plt.scatter(df_canada["CO2"], df_canada["GDP"], color='#F4FF07',
                label='Canada')
    plt.scatter(df_china["CO2"], df_china["GDP"], color='#020551',
                label='China')
    plt.scatter(df_australia["CO2"], df_australia["GDP"], color='#F4FF07',
                label='Australia')
    plt.scatter(df_brazil["CO2"], df_brazil["GDP"], color='#020551',
                label='Brazil')
    plt.scatter(df_germany["CO2"], df_germany["GDP"], color='#F4FF07',
                label='Germany')
    plt.scatter(df_japan["CO2"], df_japan["GDP"], color='#F4FF07',
                label='Japan')

    # rescale and show cluster centres
    scen = ct.backscale(cen, df_min, df_max)
    xc = scen[:, 0]
    yc = scen[:, 1]

    # plot scatter plot
    plt.scatter(xc, yc, c="k", marker="d", s=80)

    # Add country names near the data points
    offset = 0.1  # offset the text placement for better visibility
    for x, y, country in zip(dfo["CO2"], dfo["GDP"], dfo["Country Name"]):
        if country in ["Canada", "China", "Australia", "Brazil",
                       "Germany", "Japan"]:
            plt.text(x + offset, y + offset, country,
                     fontsize=11, color="black")

    # Create a legend
    legend_labels = ['Cluster {}'.format(i+1) for i in range(nc)]
    plt.legend(handles=scatter.legend_elements()[0],
               labels=legend_labels, loc="upper right")

    # Rotate y-axis label
    plt.yticks(rotation=45)

    # add legend and title
    plt.xlabel("CO2 (MT per capita)", color="black", fontweight='bold')
    plt.ylabel("GDP per capita (current US$)", color="black",
               fontweight='bold')
    plt.title("CO2 Emissions vs. GDP - 1990", color="black",
              fontweight='bold', y=1.02)
    plt.show()


def plot_normalized_cr(df, n, country_names):
    """This function returns normalized scale cluster"""

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
    for cluster1, countries in cluster_country_map.items():
        print(f"Cluster {cluster1}:")
        print(countries)

    # plot the figure
    plt.figure(figsize=(6.0, 6.0))

    # scatter plot with colours selected using the cluster numbers
    plt.scatter(df["CO2"], df["Renewable energy"], c=labels, cmap="tab10")

    # show cluster centres
    xc = cen[:, 0]
    yc = cen[:, 1]

    # plot the scatter plot
    plt.scatter(xc, yc, c="k", marker="d", s=80)

    # add title, labels, and legends
    plt.xlabel("CO2")
    plt.ylabel("Renewable energy consumption")
    plt.title("CO2 vs Renew. Energy Consumption")

    # show the plot
    plt.show()


def plot_original_scale_cr(dfn, dfo, df_min, df_max, n):
    """This function returns original scale cluster for co2 and renew.
    energy"""

    # number of cluster centres
    nc = n

    # fit normalized data to kmeans algorithm
    kmeans = cluster.KMeans(n_clusters=nc)
    kmeans.fit(dfn)

    # extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_

    # Calculate cluster statistics
    for i in range(nc):
        cluster_data = dfo.loc[labels == i, ["CO2", "Renewable energy"]]
        cluster_density = len(cluster_data)
        cluster_mean = cluster_data.mean()
        cluster_median = cluster_data.median()
        cluster_std = cluster_data.std()

        print(f"CR 2019 Cluster {i+1}:")
        print("Density:", cluster_density)
        print("Mean:", cluster_mean)
        print("Median:", cluster_median)
        print("Standard Deviation:", cluster_std)
        print()

    # plot the figure
    plt.figure(figsize=(6.0, 6.0))

    # Create separate dataframes for countries
    df_canada = dfo[dfo["Country Name"] == "Canada"]
    df_china = dfo[dfo["Country Name"] == "China"]
    df_sl = dfo[dfo["Country Name"] == "Sri Lanka"]
    df_fin = dfo[dfo["Country Name"] == "Finland"]
    df_al = dfo[dfo["Country Name"] == "Albania"]

    # plot the scatter plot
    scatter = plt.scatter(dfo["CO2"],
                          dfo["Renewable energy"],
                          c=labels,
                          cmap="Set3")
    plt.scatter(df_canada["CO2"],
                df_canada["Renewable energy"],
                color='#FFD142',
                label='Canada')
    plt.scatter(df_china["CO2"],
                df_china["Renewable energy"],
                color='#FFD142',
                label='China')
    plt.scatter(df_sl["CO2"],
                df_sl["Renewable energy"],
                color='#0FEDCF',
                label='Sri Lanka')
    plt.scatter(df_fin["CO2"],
                df_fin["Renewable energy"],
                color='#0FEDCF',
                label='Finland')
    plt.scatter(df_al["CO2"],
                df_al["Renewable energy"],
                color='#0FEDCF',
                label='Albania')

    # rescale and show cluster centres
    scen = ct.backscale(cen, df_min, df_max)
    xc = scen[:, 0]
    yc = scen[:, 1]

    # plot scatter plot
    plt.scatter(xc, yc, c="k", marker="d", s=80)

    # Add country names near the data points
    offset = 0.1  # offset the text placement for better visibility
    for x, y, country in zip(dfo["CO2"], dfo["Renewable energy"],
                             dfo["Country Name"]):
        if country in ["Canada", "China", "Sri Lanka", "Finland", "Albania"]:
            plt.text(x + offset, y + offset, country,
                     fontsize=11, color="black")

    # Create a legend
    legend_labels = ['Cluster {}'.format(i+1) for i in range(nc)]
    plt.legend(handles=scatter.legend_elements()[0],
               labels=legend_labels,
               loc="upper right")

    # add labels and titles
    plt.xlabel("CO2 (MT per capita)", color="black", fontweight='bold')
    plt.ylabel("Renewable energy Consumption (% total energy)",
               color="black", fontweight='bold')
    plt.title("CO2 vs Renew. Energy Consumption - 2019",
              color="black", fontweight='bold', y=1.02)

    # show the plot
    plt.show()


def plot_original_scale_cr_90(dfn, dfo, df_min, df_max, n):
    """This function returns original scale cluster for co2 and renew.
    energy fo 1990"""

    # number of cluster centres
    nc = n

    # fit normalized data to kmeans algorithm
    kmeans = cluster.KMeans(n_clusters=nc)
    kmeans.fit(dfn)

    # extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_

    # plot the figure
    plt.figure(figsize=(6.0, 6.0))

    # Create separate dataframes for countries
    df_canada = dfo[dfo["Country Name"] == "Canada"]
    df_china = dfo[dfo["Country Name"] == "China"]
    df_sl = dfo[dfo["Country Name"] == "Sri Lanka"]
    df_fin = dfo[dfo["Country Name"] == "Finland"]
    df_al = dfo[dfo["Country Name"] == "Albania"]

    # plot the scatter plot
    scatter = plt.scatter(dfo["CO2"],
                          dfo["Renewable energy"],
                          c=labels,
                          cmap="Set3")
    plt.scatter(df_canada["CO2"], df_canada["Renewable energy"],
                color='#020551', label='Canada')
    plt.scatter(df_china["CO2"], df_china["Renewable energy"],
                color='#020551', label='China')
    plt.scatter(df_sl["CO2"], df_sl["Renewable energy"],
                color='#F4FF07', label='Sri Lanka')
    plt.scatter(df_fin["CO2"], df_fin["Renewable energy"],
                color='#020551', label='Finland')
    plt.scatter(df_al["CO2"], df_al["Renewable energy"],
                color='#020551', label='Albania')

    # rescale and show cluster centres
    scen = ct.backscale(cen, df_min, df_max)
    xc = scen[:, 0]
    yc = scen[:, 1]

    # plot scatter plot
    plt.scatter(xc, yc, c="k", marker="d", s=80)

    # Add country names near the data points
    offset = 0.1  # offset the text placement for better visibility
    for x, y, country in zip(dfo["CO2"], dfo["Renewable energy"],
                             dfo["Country Name"]):
        if country in ["Canada", "China", "Sri Lanka", "Finland", "Albania"]:
            plt.text(x + offset, y + offset, country,
                     fontsize=11, color="black")

    # Create a legend
    legend_labels = ['Cluster {}'.format(i+1) for i in range(nc)]
    plt.legend(handles=scatter.legend_elements()[0],
               labels=legend_labels,
               loc="upper right")

    # add labels and titles
    plt.xlabel("CO2 (MT per capita)", color="black", fontweight='bold')
    plt.ylabel("Renewable energy Consumption (% total energy)",
               color="black", fontweight='bold')
    plt.title("CO2 vs Renew. Energy Consumption - 1990",
              color="black", fontweight='bold', y=1.02)

    # show the plot
    plt.show()


def plot_normalized_ge(df, n, country_names):
    """This function returns normalized scale cluster"""

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
    for cluster1, countries in cluster_country_map.items():
        print(f"Cluster {cluster1}:")
        print(countries)

    # plot the figure
    plt.figure(figsize=(6.0, 6.0))

    # scatter plot with colours selected using the cluster numbers
    plt.scatter(df["GDP"], df["Energy use"], c=labels, cmap="Set3")

    # show cluster centres
    xc = cen[:, 0]
    yc = cen[:, 1]

    # plot the scatter plot
    plt.scatter(xc, yc, c="k", marker="d", s=80)

    # add title, labels, and legends
    plt.xlabel("GDP")
    plt.ylabel("Energy Consumption")
    plt.title("GDP vs Energy Consumption")

    # show the plot
    plt.show()


def plot_original_scale_ge(dfn, dfo, df_min, df_max, n):
    """This function returns original scale cluster """

    # number of cluster centres
    nc = n

    # fit normalized data to kmeans algorithm
    kmeans = cluster.KMeans(n_clusters=nc)
    kmeans.fit(dfn)

    # extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_

    # Calculate cluster statistics
    for i in range(nc):
        cluster_data = dfo.loc[labels == i, ["GDP", "Energy use"]]
        cluster_density = len(cluster_data)
        cluster_mean = cluster_data.mean()
        cluster_median = cluster_data.median()
        cluster_std = cluster_data.std()
        print(f"Cluster {i+1}:")
        print("Density:", cluster_density)
        print("Mean:", cluster_mean)
        print("Median:", cluster_median)
        print("Standard Deviation:", cluster_std)
        print()

    # plot the figure
    plt.figure(figsize=(6.0, 6.0))

    # Create separate dataframes for countries
    df_canada = dfo[dfo["Country Name"] == "Canada"]
    df_china = dfo[dfo["Country Name"] == "China"]
    df_australia = dfo[dfo["Country Name"] == "Australia"]
    df_india = dfo[dfo["Country Name"] == "India"]
    df_qatar = dfo[dfo["Country Name"] == "Qatar"]
    df_kuwait = dfo[dfo["Country Name"] == "Bahrain"]

    # plot the scatter plot
    scatter = plt.scatter(dfo["GDP"],
                          dfo["Energy use"],
                          c=labels,
                          cmap="Set3")
    plt.scatter(df_canada["GDP"], df_canada["Energy use"],
                color='#0E5300', label='Canada')
    plt.scatter(df_china["GDP"], df_china["Energy use"],
                color='#020551', label='China')
    plt.scatter(df_australia["GDP"], df_australia["Energy use"],
                color='#0E5300', label='Australia')
    plt.scatter(df_india["GDP"], df_india["Energy use"],
                color='#020551', label='India')
    plt.scatter(df_qatar["GDP"], df_qatar["Energy use"],
                color='#F4FF07', label='Qatar')
    plt.scatter(df_kuwait["GDP"], df_kuwait["Energy use"],
                color='#F4FF07', label='Bahrain')

    # rescale and show cluster centres
    scen = ct.backscale(cen, df_min, df_max)
    xc = scen[:, 0]
    yc = scen[:, 1]

    # plot scatter plot
    plt.scatter(xc, yc, c="k", marker="d", s=80)

    # Add country names near the data points
    offset = 0.1  # offset the text placement for better visibility
    for x, y, country in zip(dfo["GDP"], dfo["Energy use"],
                             dfo["Country Name"]):
        if country in ["Canada", "China", "Australia", "India",
                       "Qatar", "Bahrain"]:
            plt.text(x + offset, y + offset, country, fontsize=11,
                     color="black")

    # Create a legend
    legend_labels = ['Cluster {}'.format(i+1) for i in range(nc)]
    plt.legend(handles=scatter.legend_elements()[0],
               labels=legend_labels,
               loc="upper left")

    # Rotate y-axis label
    plt.yticks(rotation=45)

    # add title and labels
    plt.xlabel("GDP per capita (current US$)", color="black",
               fontweight='bold')
    plt.ylabel("Energy Use per capita (kg of oil)", color="black",
               fontweight='bold')
    plt.title("GDP vs Energy Consumption - 2013", color="black",
              fontweight='bold', y=1.02)

    # show the plot
    plt.show()


def plot_original_scale_ge_90(dfn, dfo, df_min, df_max, n):
    """This function returns original scale cluster for 1990"""

    # number of cluster centres
    nc = n

    # fit normalized data to kmeans algorithm
    kmeans = cluster.KMeans(n_clusters=nc)
    kmeans.fit(dfn)

    # extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_

    # plot the figure
    plt.figure(figsize=(6.0, 6.0))

    # Create separate dataframes for countries
    df_canada = dfo[dfo["Country Name"] == "Canada"]
    df_china = dfo[dfo["Country Name"] == "China"]
    df_australia = dfo[dfo["Country Name"] == "Australia"]
    df_india = dfo[dfo["Country Name"] == "India"]
    df_qatar = dfo[dfo["Country Name"] == "Qatar"]
    df_kuwait = dfo[dfo["Country Name"] == "Bahrain"]

    # plot the scatter plot
    scatter = plt.scatter(dfo["GDP"],
                          dfo["Energy use"],
                          c=labels,
                          cmap="Set3")
    plt.scatter(df_canada["GDP"], df_canada["Energy use"],
                color='#F4FF07', label='Canada')
    plt.scatter(df_china["GDP"], df_china["Energy use"],
                color='#020551', label='China')
    plt.scatter(df_australia["GDP"], df_australia["Energy use"],
                color='#F4FF07', label='Australia')
    plt.scatter(df_india["GDP"], df_india["Energy use"],
                color='#020551', label='India')
    plt.scatter(df_qatar["GDP"], df_qatar["Energy use"],
                color='#F4FF07', label='Qatar')
    plt.scatter(df_kuwait["GDP"], df_kuwait["Energy use"],
                color='#F4FF07', label='Bahrain')

    # rescale and show cluster centres
    scen = ct.backscale(cen, df_min, df_max)
    xc = scen[:, 0]
    yc = scen[:, 1]

    # plot scatter plot
    plt.scatter(xc, yc, c="k", marker="d", s=80)

    # Add country names near the data points
    offset = 0.1  # offset the text placement for better visibility
    for x, y, country in zip(dfo["GDP"], dfo["Energy use"],
                             dfo["Country Name"]):
        if country in ["Canada", "China", "Australia", "India",
                       "Qatar", "Bahrain"]:
            plt.text(x + offset, y + offset, country, fontsize=11,
                     color="black")

    # Create a legend
    legend_labels = ['Cluster {}'.format(i+1) for i in range(nc)]
    plt.legend(handles=scatter.legend_elements()[0],
               labels=legend_labels,
               loc="upper right")

    # Rotate y-axis label
    plt.yticks(rotation=45)

    # add title and labels
    plt.xlabel("GDP per capita (current US$)", color="black",
               fontweight='bold')
    plt.ylabel("Energy Use per capita (kg of oil)", color="black",
               fontweight='bold')
    plt.title("GDP vs Energy Consumption - 1990", color="black",
              fontweight='bold', y=1.02)

    # show the plot
    plt.show()


def melt_to_one_year(df):
    """this function transform multiple year columns into one year column"""

    # transform years columns to one year column
    df_new = pd.melt(df,
                     id_vars=["Country Name",
                              "Country Code",
                              "Indicator Name"
                              ],
                     value_vars=df.iloc[:, 3:-1].columns,
                     var_name="Year",
                     value_name=("Total"))

    return df_new


def exponential_growth(t, scale, growth):
    """ Calculate exp growth """

    f = scale * np.exp(growth * (t-1990))

    return f


def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth
    rate g"""

    f = n0 / (1 + np.exp(-g*(t - t0)))

    return f


def exponential_gdp(df):
    """ return best fit using exponen. growth"""

    # calculate the percentage change in GDP for each year
    pct_changes = df["Total"].pct_change()

    # calculate the average annual growth rate
    avg_growth_rate = np.mean(pct_changes[1:])

    # fit exponential growth
    popt, pcorr = opt.curve_fit(exponential_growth,
                                df["Year"],
                                df["Total"],
                                p0=[df["Total"].iloc[-1], avg_growth_rate])

    print("Fit parameter", popt)

    # create a new column in df
    df["pop_exp"] = exponential_growth(df["Year"], *popt)

    # plot another figure
    plt.figure(figsize=(6.0, 6.0))

    # plot data and the prediction
    plt.plot(df["Year"], df["Total"], label="data")
    plt.plot(df["Year"], df["pop_exp"], label="fit")

    # add legend and title
    plt.legend()
    plt.xlabel("Year", color="black",
               fontweight='bold')
    plt.ylabel("GDP per capita (current US$)", color="black",
               fontweight='bold')
    plt.title("Final fit exponential growth for GDP China", color="black",
              fontweight='bold', y=1.01)

    # show the plot
    plt.show()

    print()
    print("GDP in")
    print("2030:", exponential_growth(2030, *popt) / 1.0e6, "Mill.")
    print("2040:", exponential_growth(2040, *popt) / 1.0e6, "Mill.")
    print("2050:", exponential_growth(2050, *popt) / 1.0e6, "Mill.")


def logistic_gdp(df):
    """return best fit using logistic function"""

    # extract the year and total GDP columns as numpy arrays
    x = df['Year'].values.astype(int)
    y = df['Total'].values

    # define the initial guess for the parameters (L, k, x0)
    p0 = [max(y), 1, np.median(x)]

    # fit the logistic model to the data
    params, covar = opt.curve_fit(logistic, x, y, p0)

    # add column for prediction
    df["pop_logistics"] = logistic(df["Year"], *params)

    print("GDP in")
    print("2030:", logistic(2030, *params) / 1.0e6, "Mill.")
    print("2040:", logistic(2040, *params) / 1.0e6, "Mill.")
    print("2050:", logistic(2050, *params) / 1.0e6, "Mill.")

    print("Fit parameter", params)
    # extract variances and calculate sigmas
    sigmas = np.sqrt(np.diag(covar))

    # create extended year range
    years = np.arange(1960, 2041)

    # call function to calculate upper and lower limits with extrapolation
    lower, upper = err.err_ranges(years, logistic, params, sigmas)

    # start plotting
    plt.figure(figsize=(6.0, 6.0))

    # plot the data and fitted line with errors
    plt.plot(df["Year"], df["Total"], label="Actual data")
    plt.plot(df["Year"], df["pop_logistics"], label="Logisic fit")

    # plot error ranges with transparency
    plt.fill_between(years, lower, upper, alpha=0.5, color='#B5EAD7')

    # add legend and title
    plt.legend(loc="upper left")
    plt.xlabel("Year", color="black",
               fontweight='bold')
    plt.ylabel("GDP per capita (current US$)", color="black",
               fontweight='bold')
    plt.title("GDP Prediction of Australia", color="black",
              fontweight='bold', y=1.01)

    print(logistic(2030, *params))
    print(err.err_ranges(2030, logistic, params, sigmas))

    # assuming symmetrie estimate sigma
    gdp2030 = logistic(2030, *params)

    # calculate pred for 2030 with lower and upper range
    low, up = err.err_ranges(2030, logistic, params, sigmas)
    sig = np.abs(up-low)/(2.0)
    print()

    print("GDP 2030", gdp2030, "+/-", sig)


    plt.text(1990, gdp2030, f"GDP 2030: {gdp2030:.2f} +/- {sig:.2f}")

    # plot the figure
    plt.show()

    return


def logistic_co2(df):
    """return best fit using logistic function"""

    # extract the year and total GDP columns as numpy arrays
    x = df['Year'].values.astype(int)
    y = df['Total'].values

    # define the initial guess for the parameters (L, k, x0)
    p0 = [max(y), 1, np.median(x)]

    # fit the logistic model to the data
    params, covar = opt.curve_fit(logistic, x, y, p0)

    # add column for prediction
    df["pop_logistics"] = logistic(df["Year"], *params)

    print("Fit parameter", params)
    # extract variances and calculate sigmas
    sigmas = np.sqrt(np.diag(covar))

    # create extended year range
    years = np.arange(1990, 2041)

    # call function to calculate upper and lower limits with extrapolation
    lower, upper = err.err_ranges(years, logistic, params, sigmas)

    # start plotting
    plt.figure(figsize=(6.0, 6.0))

    # plot the data and fitted line with errors
    plt.plot(df["Year"], df["Total"], label="Actual data")
    plt.plot(df["Year"], df["pop_logistics"], label="Logisic fit")

    # plot error ranges with transparency
    plt.fill_between(years, lower, upper, alpha=0.5, color='#EECEFD')

    # add legend and title
    plt.legend(loc="upper left")
    plt.xlabel("Year", color="black",
               fontweight='bold')
    plt.ylabel("CO2 per capita (MT)", color="black",
               fontweight='bold')
    plt.title("CO2 emission Prediction of Sweden (MT)",
              color="black",
              fontweight='bold', y=1.01)

    print(logistic(2030, *params))
    print(err.err_ranges(2030, logistic, params, sigmas))

    # assuming symmetrie estimate sigma
    co22030 = logistic(2030, *params)

    # calculate pred for 2030 with lower and upper range
    low, up = err.err_ranges(2030, logistic, params, sigmas)
    sig = np.abs(up-low)/(2.0)
    print()

    print("CO2 2030", co22030, "+/-", sig)


    plt.text(1990, co22030, f"CO2 2030: {co22030:.2f} +/- {sig:.2f}")

    # plot the figure
    plt.show()

    return



def logistic_en(df):
    """return best fit using logistic function"""

    # extract the year and total GDP columns as numpy arrays
    x = df['Year'].values.astype(int)
    y = df['Total'].values

    # define the initial guess for the parameters (L, k, x0)
    p0 = [max(y), 1, np.median(x)]

    # fit the logistic model to the data
    params, covar = opt.curve_fit(logistic, x, y, p0)

    # add column for prediction
    df["pop_logistics"] = logistic(df["Year"], *params)

    print("Fit parameter", params)
    # extract variances and calculate sigmas
    sigmas = np.sqrt(np.diag(covar))

    # create extended year range
    years = np.arange(1970, 2041)

    # call function to calculate upper and lower limits with extrapolation
    lower, upper = err.err_ranges(years, logistic, params, sigmas)

    # start plotting
    plt.figure(figsize=(6.0, 6.0))

    # plot the data and fitted line with errors
    plt.plot(df["Year"], df["Total"], label="Actual data")
    plt.plot(df["Year"], df["pop_logistics"], label="Logisic fit")

    # plot error ranges with transparency
    plt.fill_between(years, lower, upper, alpha=0.5, color='#BBF2BB')

    # add legend and title
    plt.legend(loc="lower right")
    plt.xlabel("Year", color="black",
               fontweight='bold')
    plt.ylabel("Energy Consumption (kg of oil)", color="black",
               fontweight='bold')
    plt.title("Energy Consumption of Bahrain (kg of oil)",
              color="black",
              fontweight='bold', y=1.01)

    print(logistic(2030, *params))
    print(err.err_ranges(2030, logistic, params, sigmas))

    # assuming symmetrie estimate sigma
    en2030 = logistic(2030, *params)

    # calculate pred for 2030 with lower and upper range
    low, up = err.err_ranges(2030, logistic, params, sigmas)
    sig = np.abs(up-low)/(2.0)
    print()

    print("Energy Consumption 2030", en2030, "+/-", sig)

    plt.text(1970, en2030-0.5, f"2030: {en2030:.2f} +/- {sig:.2f}")

    # plot the figure
    plt.show()

    return


# =============================================================================
# Main program
# =============================================================================


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

# get the merged data for co2 and gdp for 2019, 1990
df_2019_cg = merge_data(df_co2_year, df_gdp_year, "2019")
df_1990_cg = merge_data(df_co2_year, df_gdp_year, "1990")

# explore merged dataframe
print(df_2019_cg.describe())

# rename columns
df_2019_cg = df_2019_cg.rename(columns={"2019_x": "CO2", "2019_y": "GDP"})
df_1990_cg = df_1990_cg.rename(columns={"1990_x": "CO2", "1990_y": "GDP"})

# plot the scatter matrix
pd.plotting.scatter_matrix(df_2019_cg, figsize=(12, 12), s=20, alpha=0.8)
pd.plotting.scatter_matrix(df_1990_cg, figsize=(12, 12), s=20, alpha=0.8)

# calculate the correlation
print(df_2019_cg.corr())
print(df_1990_cg.corr())

# copy dataframe to another
df_co2_gdp_2019 = df_2019_cg[["CO2", "GDP"]].copy()
df_co2_gdp_1990 = df_1990_cg[["CO2", "GDP"]].copy()

# normalise the dataframe
df_co2_gdp_2019, df_min1, df_max1 = ct.scaler(df_co2_gdp_2019)
df_co2_gdp_1990, df_min4, df_max4 = ct.scaler(df_co2_gdp_1990)

# get the merged data for co2 and renewable energy for 2019
df_2019_cr = merge_data(df_co2_year, df_rec_year, "2019")
df_1990_cr = merge_data(df_co2_year, df_rec_year, "1990")

# explore merged dataframe
print(df_2019_cr.describe())
print(df_1990_cr.describe())


# rename columns
df_2019_cr = df_2019_cr.rename(columns={"2019_x": "CO2",
                                        "2019_y": "Renewable energy"})
df_1990_cr = df_1990_cr.rename(columns={"1990_x": "CO2",
                                        "1990_y": "Renewable energy"})

# plot the scatter matrix
pd.plotting.scatter_matrix(df_2019_cr, figsize=(12, 12), s=20, alpha=0.8)
pd.plotting.scatter_matrix(df_1990_cr, figsize=(12, 12), s=20, alpha=0.8)

# calculate the correlation
print(df_2019_cr.corr())
print(df_1990_cr.corr())

# copy dataframe to another
df_co2_rec_2019 = df_2019_cr[["CO2", "Renewable energy"]].copy()
df_co2_rec_1990 = df_1990_cr[["CO2", "Renewable energy"]].copy()

# normalise the dataframe
df_co2_rec_2019, df_min2, df_max2 = ct.scaler(df_co2_rec_2019)
df_co2_rec_1990, df_min5, df_max5 = ct.scaler(df_co2_rec_1990)

# get the merged data for co2 and renewable energy for 2013
df_2013_ge = merge_data(df_gdp_year, df_energy_year, "2013")
df_1990_ge = merge_data(df_gdp_year, df_energy_year, "1990")

# explore the merged dataframe
print(df_2013_ge.describe())
print(df_1990_ge.describe())

# rename columns
df_2013_ge = df_2013_ge.rename(columns={"2013_x": "GDP",
                                        "2013_y": "Energy use"})
df_1990_ge = df_1990_ge.rename(columns={"1990_x": "GDP",
                                        "1990_y": "Energy use"})

# plot the scatter matrix
pd.plotting.scatter_matrix(df_2013_ge, figsize=(12, 12), s=20, alpha=0.8)
pd.plotting.scatter_matrix(df_1990_ge, figsize=(12, 12), s=20, alpha=0.8)

# calculate the correlation
print(df_2013_ge.corr())
print(df_1990_ge.corr())

# copy dataframe to another
df_gdp_en_2019 = df_2013_ge[["GDP", "Energy use"]].copy()
df_gdp_en_1990 = df_1990_ge[["GDP", "Energy use"]].copy()

# normalise the dataframe
df_gdp_en_2019, df_min3, df_max3 = ct.scaler(df_gdp_en_2019)
df_gdp_en_1990, df_min6, df_max6 = ct.scaler(df_gdp_en_1990)

# calculate silhoutte score
print("Score for CO2/GDP 2019:")
calculate_silhoutte_score(df_co2_gdp_2019)
print("Score for CO2/REC 2019:")
calculate_silhoutte_score(df_co2_rec_2019)
print("Score for GDP/EC 2013:")
calculate_silhoutte_score(df_gdp_en_2019)
print("Score for CO2/GDP 1990:")
calculate_silhoutte_score(df_co2_gdp_1990)
print("Score for CO2/REC 1990:")
calculate_silhoutte_score(df_co2_rec_1990)
print("Score for GDP/EC 1990:")
calculate_silhoutte_score(df_gdp_en_1990)


# get country names
country_names_cg = df_2019_cg["Country Name"].tolist()
country_names_cr = df_2019_cr["Country Name"].tolist()
country_names_ge = df_2013_ge["Country Name"].tolist()
country_names_cg_90 = df_1990_cg["Country Name"].tolist()
country_names_cr_90 = df_1990_cr["Country Name"].tolist()
country_names_ge_90 = df_1990_ge["Country Name"].tolist()

# call function to plot normalized cluster
print("")
print("Clusters for CO2/GDP 2019:")
plot_normalized_cg(df_co2_gdp_2019, 3, country_names_cg)
print("")
print("Clusters for CO2/REC 2019:")
plot_normalized_cr(df_co2_rec_2019, 2, country_names_cr)
print("")
print("Clusters for GDP/EC 2013:")
plot_normalized_ge(df_gdp_en_2019, 3, country_names_ge)
print("")
print("Clusters for CO2/GDP 1990:")
plot_normalized_cg(df_co2_gdp_1990, 2, country_names_cg_90)
print("")
print("Clusters for CO2/REC 1990:")
plot_normalized_cr(df_co2_rec_1990, 2, country_names_cr_90)
print("")
print("Clusters for GDP/EC 1990:")
plot_normalized_ge(df_gdp_en_1990, 2, country_names_ge_90)

# call function to plot original scale cluster
plot_original_scale_cg(df_co2_gdp_2019, df_2019_cg, df_min1, df_max1, 2)
plot_original_scale_cr(df_co2_rec_2019, df_2019_cr, df_min2, df_max2, 2)
plot_original_scale_ge(df_gdp_en_2019, df_2013_ge, df_min3, df_max3, 3)
plot_original_scale_cg_1990(df_co2_gdp_1990, df_1990_cg, df_min4, df_max4, 2)
plot_original_scale_cr_90(df_co2_rec_1990, df_1990_cr, df_min5, df_max5, 2)
plot_original_scale_ge_90(df_gdp_en_1990, df_1990_ge, df_min6, df_max6, 2)

# transform the df_year dataframe seperate years columns into one year column
df_gdp_year_pivot = melt_to_one_year(df_gdp_year)
df_co2_year_pivot = melt_to_one_year(df_co2_year)
df_rec_year_pivot = melt_to_one_year(df_rec_year)
df_en_year_pivot = melt_to_one_year(df_energy_year)

# extract data for china
df_gdp_china = df_gdp_year_pivot[df_gdp_year_pivot["Country Name"] ==
                               "Australia"]

df_co2_china = df_co2_year_pivot[df_co2_year_pivot["Country Name"] ==
                               "Sweden"]

df_en_china = df_en_year_pivot[df_en_year_pivot["Country Name"] ==
                               "Bahrain"]

# drop NaNs
df_gdp_china = df_gdp_china.dropna(axis=0)
df_co2_china = df_co2_china.dropna(axis=0)
df_en_china = df_en_china.dropna(axis=0)

# convert year into int data type
df_gdp_china['Year'] = df_gdp_china['Year'].astype(int)
df_co2_china['Year'] = df_co2_china['Year'].astype(int)
df_en_china['Year'] = df_en_china['Year'].astype(int)

# explore the new dataframe
print(df_gdp_china.info())

# call exp function to fit data with exp function
exponential_gdp(df_gdp_china)

# call logistic function to fit data with logistic function
logistic_gdp(df_gdp_china)
logistic_co2(df_co2_china)
logistic_en(df_en_china)


