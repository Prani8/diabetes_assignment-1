#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:34:43 2024

@author: praneet sivakumar
"""

# Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway
import statsmodels.api as sm

# Load and inspect the dataset
diabetes_data = pd.read_csv('diabetes_dataset.csv')
print(diabetes_data.head())
print(diabetes_data.info())
print(diabetes_data.describe())
print("Missing values per column:\n", diabetes_data.isnull().sum())

# Replace zero values with NaN to mark missing data in specific columns


def replace_zeros_with_nan(data, columns):
    """
    Replaces zero values with NaN in selected columns to indicate missing data for more accurate analysis.
    """
    data = data.copy()
    data[columns] = data[columns].replace(0, np.nan)
    return data


columns_to_replace = ['Glucose', 'BloodPressure',
                      'SkinThickness', 'Insulin', 'BMI', 'Age']
cleaned_data = replace_zeros_with_nan(diabetes_data, columns_to_replace)

# Fill missing values with median values


def fill_missing_values_with_median(data, columns):
    """
    Fills missing values (NaN) in specified columns with the median value of each column, preserving data consistency.
    """
    for column in columns:
        median_value = data[column].median()
        data[column].fillna(median_value, inplace=True)
    return data


# Fill missing values with median
final_cleaned_data = fill_missing_values_with_median(
    cleaned_data, columns_to_replace)
print("Data after filling NaN values with medians:\n",
      final_cleaned_data.head())

figsize = (10, 6)

# Correlation Matrix Heatmap


def plot_correlation_matrix(data, cmap='GnBu'):
    """
    Creates a heatmap to show the correlation matrix of features in the dataset, highlighting relationships between variables.
    """
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap=cmap, linewidths=0.5)
    plt.title('Correlation Matrix of Diabetes Features', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig("heatmap_feature_correlation.png")
    plt.show()


# Call the function to display the heatmap
plot_correlation_matrix(final_cleaned_data, cmap='GnBu')

# BMI Levels Histogram


def plot_bmi_distribution(data, column_name, title="BMI Distribution",
                          xlabel="BMI", ylabel="Frequency"):
    """
    Plots a histogram showing the distribution of BMI values in the dataset to illustrate frequency and variation in BMI.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=20, color='skyblue', edgecolor='black')
    plt.title(f'{title}', fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True)
    plt.savefig("histogram_bmi_distribution.png")
    plt.show()


# Call the function to plot BMI distribution
plot_bmi_distribution(final_cleaned_data['BMI'], column_name='BMI')

# Blood Pressure vs Glucose Scatter Plot


def plot_blood_pressure_vs_glucose(data, x, y,
                                   title="Blood Pressure vs Glucose",
                                   xlabel="Glucose Levels",
                                   ylabel="Blood Pressure"):
    """
    Displays a scatter plot to explore the relationship between glucose levels and blood pressure for early insights.
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(x=x, y=y, data=data.head(30), alpha=0.7)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True)
    plt.savefig("scatter_plot_blood_pressure_vs_glucose.png")
    plt.show()
    print(f"Interpretation: Analyze the relationship between {x} and {y}.")


# Call the function to plot blood pressure vs. glucose levels
plot_blood_pressure_vs_glucose(
    final_cleaned_data, x='Glucose', y='BloodPressure')

# Insulin Levels by Age Group Line Chart


def plot_insulin_by_age_group(data, x_column, y_column, hue_column=None,
                              title="Insulin Levels by Age Group",
                              xlabel="Age", ylabel="Insulin Levels"):
    """
    Creates a line chart to visualize insulin levels across different age groups, providing insights on insulin trends with age.
    """
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=x_column, y=y_column, hue=hue_column,
                 data=data, marker="o", palette='Set2')
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True)
    plt.savefig("line_chart_age_vs_insulin_by_age_group.png")
    plt.show()


# Add an age group column for age group distinction
final_cleaned_data['Age Group'] = pd.cut(final_cleaned_data['Age'], bins=[
                                         0, 30, 50, 100], labels=['Young',
                                                                  'Middle-aged',
                                                                  'Senior'])

# Call the function to plot insulin levels by age group
plot_insulin_by_age_group(
    final_cleaned_data, x_column='Age', y_column='Insulin',
    hue_column='Age Group')

# Age Distribution by Diabetes Outcome Box Plot


def plot_age_distribution_by_diabetes(data, x, y, title="Age Distribution by Diabetes Outcome", xlabel="Diabetes Outcome", ylabel="Age"):
    """
    Displays a box plot comparing the distribution of ages among individuals with and without diabetes to show age-related patterns.
    """
    plt.figure(figsize=figsize)
    sns.boxplot(x=x, y=y, data=data, palette='Set2')
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True)
    plt.savefig("box_plot_age_distribution_by_outcome.png")
    plt.show()
    print(
        f"Interpretation: Compare {y} distribution across categories of {x}.")


# Call the function to plot age distribution by diabetes outcome
plot_age_distribution_by_diabetes(final_cleaned_data, x='Outcome', y='Age')
