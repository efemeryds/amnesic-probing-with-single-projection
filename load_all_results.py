import pickle
import pandas as pd

def load_all_results(file_path, name):
    """
    Load the all_results.pkl file.

    Args:
        file_path (str): Path to the all_results.pkl file.
        name (str): A name to identify the results.

    Returns:
        dict: Loaded results from the pickle file.
    """
    try:
        with open(file_path, 'rb') as file:
            all_results = pickle.load(file)
        print(f"Successfully loaded {name}")
        print(all_results)
        print("-----------------")
        return all_results
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check the path and try again.")
    except pickle.UnpicklingError:
        print("Error: Could not unpickle the file. Ensure it's a valid pickle file.")
    return None

def process_results_to_dataframe(result_dict, name):
    """
    Process a result dictionary into a structured DataFrame.

    Args:
        result_dict (dict): Dictionary of results.
        name (str): Identifier for the data source.

    Returns:
        pd.DataFrame: DataFrame of structured results.
    """
    if not result_dict:
        return pd.DataFrame()  # Return an empty DataFrame for missing results

    # Create DataFrame from the dictionary
    data = {
        "source": [name] * len(result_dict),
        "metric": list(result_dict.keys()),
        "value": list(result_dict.values())
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    # File paths and their respective names
    target = "removed_leace"
    file_paths = {
        "cpos_masked": f"results/100k_batches_SGD_stable/masked/cpos/{target}/selectivity_results.pkl",
        "dep_masked": f"results/100k_batches_SGD_stable/masked/dep/{target}/selectivity_results.pkl",
        "fpos_masked": f"results/100k_batches_SGD_stable/masked/fpos/{target}/selectivity_results.pkl",
        "cpos_normal": f"results/100k_batches_SGD_stable/normal/cpos/{target}/selectivity_results.pkl",
        "dep_normal": f"results/100k_batches_SGD_stable/normal/dep/{target}/selectivity_results.pkl",
        "fpos_normal": f"results/100k_batches_SGD_stable/normal/fpos/{target}/selectivity_results.pkl"
    }

    # Process all results into a combined DataFrame
    all_dataframes = []
    for name, path in file_paths.items():
        results = load_all_results(path, name)
        df = process_results_to_dataframe(results, name)
        all_dataframes.append(df)

    # Combine all DataFrames
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Save to CSV
    # combined_df.to_csv("all_results_combined.csv", index=False)
    # combined_df.to_excel("all_results_combined.xlsx", index=False)
    print("Saved combined results to all_results_combined.csv")
