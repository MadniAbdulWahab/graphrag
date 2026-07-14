import pandas as pd
from datasets import load_dataset

def load_huggingface_dataset(dataset: str, split='validation'):
    """
    Load the specified dataset from huggingface:

    Args:
        - dataset: HuggingFace dataset name.
    Returns:
        - Pandas DataFrame containing the dataset split.
     
    """
    ds = load_dataset(dataset)
    
    return to_dataframe(ds, split) 

def to_dataframe(dataset, split="validation"):
    """
    Convert the specified split of the dataset to a pandas DataFrame.
    Args:
        - dataset: Loaded Hugging Face dataset.
        - split: Dataset split to convert (e.g., 'train', 'validation').
    Returns:
        - Pandas DataFrame containing the dataset split.
    """
    # Convert the specified split to a pandas DataFrame
    df = dataset[split].to_pandas()
    return df


def get_context_merged_datset(dataset_name="rajpurkar/squad") -> pd.DataFrame:
    loaded_dataframe = load_huggingface_dataset(dataset=dataset_name)
    
    # df = to_dataframe(loaded_dataset)
    
    # print
    merged_df = (
    loaded_dataframe.groupby('context', as_index=False)
        .agg({
            'id': lambda x: list(x),          # Combine IDs into a list
            'title': 'first',                 # Take the first title (assuming all titles are the same for the same context)
            'question': list,                 # Combine questions into a list
            'answers': list                   # Combine answers into a list
        })
    )

    return merged_df
    
##! For debuging 
# df = get_context_merged_datset()
# print(df.head())