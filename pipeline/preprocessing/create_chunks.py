import pandas as pd
import re

def filter_html_tags(tokens):
    """
    Remove the HTML tags from the tokens list.

    Args:
        tokens (list): List of tokens in order from the context.
    """
    # Regular expression pattern to match HTML tags
    html_tag_pattern = re.compile(r'<[^>]+>')
    # Filter out tokens that match the HTML tag pattern
    filtered_tokens = [token for token in tokens if not html_tag_pattern.match(token)]
    return filtered_tokens

def create_chunks(text, chunk_size=600, overlap_size=10):
    """
    Create chunks from the input text, ensuring each chunk ends at a sentence boundary and has overlap.

    Args:
        text (str): Input text to be split into chunks.
        chunk_size (int): The maximum number of tokens (words) per chunk.
        overlap_size (int): Number of tokens to overlap between consecutive chunks.

    Returns:
        pd.DataFrame: A new DataFrame where each row contains a chunk of text with token limits and overlap.
                      Additional columns include the original row index to map chunks back to the original text.
    """
    # Tokenize the input text into words
    tokens = text.split()

    chunks = []
    original_indices = []

    i = 0
    while i < len(tokens):
        # Define the end of the current chunk (up to chunk_size tokens)
        end = min(i + chunk_size, len(tokens))
        chunk = tokens[i:end]

        # Create the chunk text
        chunk_text = " ".join(chunk)
        
        # Regular expression to find the last punctuation mark that could end a sentence
        sentence_end = re.search(r'([.!?])(?=\s|$)', chunk_text[::-1])
        
        if sentence_end:
            # Adjust the end of the chunk to the last sentence-ending punctuation
            end_offset = len(chunk_text) - sentence_end.end()
            chunk_text = chunk_text[:end_offset + 1]
        
        chunks.append(chunk_text)
        original_indices.append(i // chunk_size)  # Track the chunk index

        # Move to the next chunk start with overlap
        i = i + chunk_size - overlap_size

    # Create a DataFrame with chunks and their original indices
    df = pd.DataFrame({
        'chunk': chunks,
        'original_index': original_indices
    })
    
    return df
