import pandas as pd
import torch
import numpy as np
import glob
import re
from tqdm import tqdm
from itertools import combinations
import random
from sklearn.model_selection import train_test_split

tqdm.pandas()

# Load author references data
author_references_test = pd.read_hdf(
    "/mnt/home/amadovic/neural_author_disambiguator/author_references_nov22nd_v2.h5"
).reset_index()


# Function to extract chunk number from file path
def extract_chunk_number(file_path):
    match = re.search(r"_(\d+)\.pt$", file_path)
    return int(match.group(1)) if match else None


# Function to process embeddings from a directory
def process_embeddings(directory_path, file_pattern):
    files = glob.glob(directory_path + file_pattern)
    sorted_file_paths = sorted(files, key=extract_chunk_number)
    embeddings_list = []

    for file in sorted_file_paths:
        print(file)
        embeddings = torch.tensor(torch.load(file, map_location="cpu"))
        embeddings_list.append(embeddings)

    flattened_embeddings_list = [
        item
        for sublist in embeddings_list
        for item in (sublist if isinstance(sublist, list) else [sublist])
    ]
    concatenated_embeddings = torch.cat(flattened_embeddings_list, dim=0)
    concatenated_np_embeddings = concatenated_embeddings.detach().cpu().numpy()

    return concatenated_np_embeddings


# Process specter embeddings
specter_embeddings = process_embeddings(
    directory_path="/specter_embeddings/",
    file_pattern="author_embeddings_batch_*",
)

# Process author embeddings
author_chars2vec = np.load(
    "author_embeddings.npy"
)

# Check shapes of embeddings
specter_embeddings.shape, author_chars2vec.shape

# Assuming your data is in a DataFrame called author_references
# Create a combined list of unique ORCID identifiers from @path
unique_orcids = author_references['@path'].unique()

# Convert unique ORCIDs into a DataFrame
unique_orcids_df = pd.DataFrame(unique_orcids, columns=['orcid'])

# Split the ORCIDs into train, val, and test sets to avoid data leakage later on 
train_orcids, test_orcids = train_test_split(unique_orcids_df['orcid'], test_size=0.2, random_state=random_state)
train_orcids, val_orcids = train_test_split(train_orcids, test_size=0.2, random_state=random_state)  # 0.25 * 0.8 = 0.2

# Assign the train, val, and test sets based on both @path and @path2 ORCID groups
def assign_split(row):
    if row['@path'] in train_orcids.values:
        return 'train'
    elif row['@path'] in val_orcids.values:
        return 'val'
    else:
        return 'test'

author_references['split'] = author_references.progress_apply(assign_split, axis=1)


# Function to flatten lists
def flatten_lists(column):
    return column.apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)


# Process blocks for hard negatives
def process_blocks(group):
    data_list = []
    for (i_index, i), (j_index, j) in combinations(group.iterrows(), 2):
        data = {
            "index1": i_index,
            "@path": i["@path"],
            "abstract": i["abstract"],
            "author": i["author"],
            "index2": j_index,
            "@path2": j["@path"],
            "abstract2": j["abstract"],
            "author2": j["author"],
            "label": i["@path"] == j["@path"],
        }
        data_list.append(data)
    if data_list:
        result_df = pd.DataFrame(data_list).apply(flatten_lists)
        return result_df
    else:
        return pd.DataFrame()


group = author_references.reset_index(drop=True).groupby(['block', 'split'])
author_index_pairs = group.progress_apply(process_blocks)

# Balance the dataset
random_state = 11
sample_size_per_class = (
    author_index_pairs.groupby("label")
    .count()
    .query("label == False")["index1"]
    .values[0]
)
class_0_data = author_index_pairs[author_index_pairs["label"] == True]
class_1_data = author_index_pairs[author_index_pairs["label"] == False]

sample_class_0 = class_0_data.sample(n=sample_size_per_class, random_state=random_state)
sample_class_1 = class_1_data.sample(n=sample_size_per_class, random_state=random_state)

balanced_sample = pd.concat([sample_class_0, sample_class_1])
balanced_sample = balanced_sample.sample(frac=1).reset_index(drop=True)

# Prepare embeddings for pairs
auth1_chars2vec = []
auth2_chars2vec = []
specter1_embed = []
specter2_embed = []

for _, row in tqdm(balanced_sample.iterrows()):
    auth1_chars2vec.append(author_chars2vec[int(row["index1"])])
    auth2_chars2vec.append(author_chars2vec[int(row["index2"])])
    specter1_embed.append(specter_embeddings[int(row["index1"])])
    specter2_embed.append(specter_embeddings[int(row["index2"])])

# Convert lists to numpy arrays
author1_embed = np.concatenate(
    [auth1_chars2vec, specter1_embed], axis=1
)
author2_embed = np.concatenate(
    [auth2_chars2vec, specter2_embed], axis=1
)
