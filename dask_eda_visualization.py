import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define file paths
BASE_DATA_PATH = "data/processed/"
TRAIN_DATA_PATH = os.path.join(BASE_DATA_PATH, "gsm8k_train.parquet")
TEST_DATA_PATH = os.path.join(BASE_DATA_PATH, "gsm8k_test.parquet")

def load_and_inspect_data(file_path, dataset_name):
    """Loads a parquet dataset using Dask and prints basic info."""
    print(f"--- {dataset_name} Dataset ---")
    try:
        ddf = dd.read_parquet(file_path)
        print(f"Successfully loaded {file_path}")
        
        print("\nFirst 5 rows:")
        print(ddf.head())
        
        print("\nData types:")
        print(ddf.dtypes)
        
        print("\nShape (computed):")
        # For Dask, computing shape can be expensive if the metadata isn't available
        # We'll try to get it, but be mindful of performance on very large datasets
        try:
            num_rows = len(ddf) # This computes the number of rows
            num_cols = len(ddf.columns)
            print(f"({num_rows}, {num_cols})")
        except Exception as e:
            print(f"Could not compute shape directly: {e}")
            print("Number of columns:", len(ddf.columns))
            print("Number of partitions:", ddf.npartitions)


        print("\nMissing values (computed):")
        print(ddf.isnull().sum().compute())
        
        return ddf
    except Exception as e:
        print(f"Error loading or inspecting {file_path}: {e}")
        return None

if __name__ == "__main__":
    print("Starting Dask EDA and Visualization script...")

    # Ensure necessary libraries are installed
    print("Please ensure you have 'dask', 'pandas', 'pyarrow' or 'fastparquet', 'matplotlib', and 'seaborn' installed in your environment.")
    print("You can typically install them using: conda install dask pandas pyarrow matplotlib seaborn -c conda-forge")
    print("You can typically install them using: conda install dask pandas pyarrow matplotlib seaborn -c conda-forge")
    print("-" * 50)

    ddf_train = load_and_inspect_data(TRAIN_DATA_PATH, "Training")
    print("-" * 50)
    ddf_test = load_and_inspect_data(TEST_DATA_PATH, "Test")
    print("-" * 50)

    if ddf_train is not None:
        print("\nFurther analysis and visualization for Training data can be added here.")
        # Example: Describe numerical columns (if any)
        # numerical_cols_train = ddf_train.select_dtypes(include=['number']).columns
        # if not numerical_cols_train.empty:
        #     print("\nDescription of numerical columns (Training):")
        #     print(ddf_train[numerical_cols_train].describe().compute())
        pass

    if ddf_test is not None:
        print("\nFurther analysis and visualization for Test data can be added here.")
        # Example: Describe numerical columns (if any)
        # numerical_cols_test = ddf_test.select_dtypes(include=['number']).columns
        # if not numerical_cols_test.empty:
        #     print("\nDescription of numerical columns (Test):")
        #     print(ddf_test[numerical_cols_test].describe().compute())
        pass

    print("\nScript finished. Next steps: Add specific visualization functions based on the data structure revealed above.")
    print("For example, if you have text columns, consider word clouds or length distributions.")
    print("If you have numerical columns, consider histograms or scatter plots.")
    print("If you have categorical columns, consider bar charts.")

    # Placeholder for visualizations - we'll add these in the next step
    # def plot_example_histogram(ddf, column_name, title):
    #     plt.figure(figsize=(10, 6))
    #     # For Dask, direct plotting might require computing data first
    #     # Or using Dask-compatible plotting libraries/methods
    #     data_to_plot = ddf[column_name].compute() # Compute for plotting
    #     sns.histplot(data_to_plot, kde=True)
    #     plt.title(title)
    #     plt.xlabel(column_name)
    #     plt.ylabel("Frequency")
    #     # Ensure logs directory exists
    #     if not os.path.exists("logs"):
    #         os.makedirs("logs")
    #     plt.savefig(f"logs/{title.replace(' ', '_')}.png")
    #     print(f"Saved plot to logs/{title.replace(' ', '_')}.png")
    #     # plt.show() # plt.show() might not work well in a non-interactive SLURM environment

    # if ddf_train is not None and 'some_numerical_column' in ddf_train.columns:
    #     plot_example_histogram(ddf_train, 'some_numerical_column', 'Distribution of Some Numerical Column (Train)') 