import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
def load_and_explore():
    # Load the Iris dataset
    iris = load_iris(as_frame=True)
    df = iris.frame

    # Display the first few rows
    print("First 5 rows of the dataset:")
    print(df.head())

    # Check data types and missing values
    print("\nDataset Info:")
    print(df.info())

    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Clean the dataset (no missing values in Iris, but this is an example)
    df = df.dropna()

    return df

# Task 2: Basic Data Analysis
def basic_analysis(df):
    # Compute basic statistics
    print("\nBasic Statistics:")
    print(df.describe())

    # Group by species and compute mean of numerical columns
    print("\nMean values grouped by species:")
    grouped = df.groupby('target').mean()
    print(grouped)

# Task 3: Data Visualization
def visualize_data(df):
    # Line chart (example: cumulative sum of sepal length)
    df['cumulative_sepal_length'] = df['sepal length (cm)'].cumsum()
    plt.figure(figsize=(8, 5))
    plt.plot(df.index, df['cumulative_sepal_length'], label='Cumulative Sepal Length')
    plt.title('Cumulative Sepal Length Over Index')
    plt.xlabel('Index')
    plt.ylabel('Cumulative Sepal Length')
    plt.legend()
    plt.show()

    # Bar chart (average petal length per species)
    plt.figure(figsize=(8, 5))
    sns.barplot(x='target', y='petal length (cm)', data=df, ci=None)
    plt.title('Average Petal Length per Species')
    plt.xlabel('Species')
    plt.ylabel('Petal Length (cm)')
    plt.show()

    # Histogram (distribution of sepal width)
    plt.figure(figsize=(8, 5))
    plt.hist(df['sepal width (cm)'], bins=10, color='skyblue', edgecolor='black')
    plt.title('Distribution of Sepal Width')
    plt.xlabel('Sepal Width (cm)')
    plt.ylabel('Frequency')
    plt.show()

    # Scatter plot (sepal length vs petal length)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='target', data=df)
    plt.title('Sepal Length vs Petal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend(title='Species')
    plt.show()

# Main function
if __name__ == "__main__":
    df = load_and_explore()
    basic_analysis(df)
    visualize_data(df)