
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from scipy.optimize import curve_fit
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer


# Function to load data from a CSV file
def get_data(file_path):
    return pd.read_csv(file_path)


# Function to preprocess the data (
def preprocess_data(data):
    # Melt the data for easier manipulation
    data_melted = data.melt(id_vars=['Country Name', 'Country Code',
                            'Series Name', 'Series Code'], var_name='Year', value_name='Value')
    data_melted.replace('..', np.nan, inplace=True)

    # Group melted data to create a pivoted DataFrame
    data_pivoted = data_melted.groupby(['Country Name', 'Country Code', 'Year', 'Series Name'])[
        'Value'].mean().unstack().reset_index()

    # Pivot tables for better organization
    data_countries = data_melted.pivot_table(
        index=['Year', 'Series Name'], columns='Country Name', values='Value')
    data_years = data_melted.pivot_table(
        index=['Country Name', 'Series Name'], columns='Year', values='Value')

    # Fill missing values with the mean of each column
    data_new = data_pivoted.fillna(data_pivoted.mean())
    data_new.to_csv("CFData_new.csv")

    # Extract numeric columns for imputation and clustering
    numeric_columns = data_new.select_dtypes(include=[np.number]).columns
    numeric_data = data_new[numeric_columns]

    # Impute missing values for numeric data
    imputer = SimpleImputer(strategy='mean')
    numeric_data_imputed = pd.DataFrame(
        imputer.fit_transform(numeric_data), columns=numeric_columns)

    # Combine imputed numeric data with non-numeric data
    data_imputed = pd.concat(
        [numeric_data_imputed, data_new.select_dtypes(exclude=[np.number])], axis=1)

    # Data Normalization
    normalized_data = (
        numeric_data_imputed - numeric_data_imputed.mean()) / numeric_data_imputed.std()

    return data_imputed, normalized_data


# Function to perform clustering using KMeans
def perform_clustering(normalized_data, num_clusters=4):
    # Create KMeans model
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clustered_data = normalized_data.copy()

    # Add a 'Cluster' column to the DataFrame
    clustered_data['Cluster'] = kmeans.fit_predict(normalized_data)

    # Calculate Silhouette Score for evaluation
    silhouette_avg = silhouette_score(
        normalized_data, clustered_data['Cluster'])
    print(f"Silhouette Score: {silhouette_avg}")

    # Visualize silhouette plot
    visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
    visualizer.fit(normalized_data)
    visualizer.show()

    return clustered_data, kmeans


# Function to visualize clusters CO2 and GDP
def visualize_clusters(clustered_data, kmeans, features=['GDP growth (annual %)', 'CO2 emissions (kt)']):
    plt.figure(figsize=(10, 8))
    for cluster in clustered_data['Cluster'].unique():
        cluster_data = clustered_data[clustered_data['Cluster'] == cluster]
        plt.scatter(cluster_data[features[0]],
                    cluster_data[features[1]], label=f'Cluster {cluster}')

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
                :, 1], s=200, c='black', marker='X', label='Cluster Centers')
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title(
        f'Clustering of Countries based on "{features[0]}" and "{features[1]}"')
    plt.legend()
    plt.show()


# Additional cluster visualization Greenhouse and CO2
def visualize_clusters_custom(clustered_data, kmeans, features=['Total greenhouse gas emissions (kt of CO2 equivalent)', 'CO2 emissions (kt)']):
    plt.figure(figsize=(10, 8))
    for cluster in clustered_data['Cluster'].unique():
        cluster_data = clustered_data[clustered_data['Cluster'] == cluster]
        plt.scatter(cluster_data[features[0]],
                    cluster_data[features[1]], label=f'Cluster {cluster}')

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
                :, 1], s=200, c='black', marker='X', label='Cluster Centers')
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title(
        f'Clustering of Countries based on "{features[0]}" and "{features[1]}"')
    plt.legend()
    plt.show()


# Function to define the exponential curve
def curve(x, a, b, c):
    return a * np.exp(b * (x - x.iloc[0])) + c


# Function to fit and plot the curve 
def fit_and_plot_curve(data, country_name, x_column, y_column):
    country_data = data[data['Country Name'] == country_name]
    years = country_data[x_column].astype(int)
    emissions = country_data[y_column].astype(float)

    # Fit the curve using curve_fit
    curve_params, _ = curve_fit(curve, years, emissions, maxfev=1000)
    prediction_years = pd.Series(range(1990, 2031))
    predicted_emissions = curve(prediction_years, *curve_params)

    # Plot the actual and predicted data
    plt.figure(figsize=(10, 6))
    plt.plot(years, emissions, 'o-',
             label=f'{country_name} Actual Data', color='green')
    plt.plot(prediction_years, predicted_emissions, '.',
             label=f'{country_name} Predicted Data (1990-2030)', color='red')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'{y_column} of {country_name} with Curve Fit (1990-2030)')
    plt.legend()
    plt.grid(True)
    plt.show()


# Function to fit and plot the curve with confidence range 
def fit_and_plot_curve_confidence_range(data, country_name, x_column, y_column):
    country_data = data[data['Country Name'] == country_name]
    years = country_data[x_column].astype(int)
    emissions = country_data[y_column].astype(float)

    # Fit the curve using curve_fit
    curve_params, cov_matrix = curve_fit(curve, years, emissions, maxfev=1000)
    prediction_years = pd.Series(range(1990, 2031))
    predicted_emissions = curve(prediction_years, *curve_params)
    
    # Estimate confidence intervals
    perr = np.sqrt(np.diag(cov_matrix))
    lower_bound = curve(prediction_years, *(curve_params - perr))
    upper_bound = curve(prediction_years, *(curve_params + perr))

    # Plot the actual data, predicted data, and confidence range
    plt.figure(figsize=(10, 6))
    plt.plot(years, emissions, 'o-',
             label=f'{country_name} Actual Data', color='green')
    plt.plot(prediction_years, predicted_emissions, '.',
             label=f'{country_name} Predicted Data (1990-2030)', color='red')
    plt.fill_between(prediction_years, lower_bound, upper_bound,
                     color='gray', alpha=0.3, label='Confidence Range')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'{y_column} of {country_name} with Curve Fit (1990-2030)')
    plt.legend()
    plt.grid(True)
    plt.show()


# Function to estimate confidence intervals for curve parameters
def err_ranges(x, a, b, c, delta_a, delta_b, delta_c):
    return a * np.exp(b * (x - x.iloc[0])) + c + delta_a * np.exp(delta_b * (x - x.iloc[0])) + delta_c


# Main function to execute the entire analysis
def main():
    # Load data
    cf_data = get_data('CFData.csv')

    # Preprocess data
    cf_data_processed, normalized_data = preprocess_data(cf_data)

    # Perform clustering
    cf_data_clustered, kmeans = perform_clustering(normalized_data)

    # Visualize clusters
    visualize_clusters(cf_data_clustered, kmeans)

    # Additional cluster visualization based on different features
    visualize_clusters_custom(cf_data_clustered, kmeans, features=[
                              'Total greenhouse gas emissions (kt of CO2 equivalent)', 'CO2 emissions (kt)'])

    # Fit and plot curves for specific countries
    fit_and_plot_curve(cf_data_processed, 'India', 'Year',
                       'Total greenhouse gas emissions (kt of CO2 equivalent)')
    fit_and_plot_curve(cf_data_processed, 'France', 'Year',
                       'Total greenhouse gas emissions (kt of CO2 equivalent)')

    # Fit and plot curves with confidence range for specific countries
    fit_and_plot_curve_confidence_range(
        cf_data_processed, 'India', 'Year', 'Total greenhouse gas emissions (kt of CO2 equivalent)')
    fit_and_plot_curve_confidence_range(
        cf_data_processed, 'France', 'Year', 'Total greenhouse gas emissions (kt of CO2 equivalent)')


if __name__ == "__main__":
    main()
