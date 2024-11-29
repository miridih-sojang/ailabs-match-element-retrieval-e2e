import pandas as pd
import numpy as np
from tqdm import tqdm

tqdm.pandas()


def evaluate(df, top_k_list=[1, 4, 5, 10, 20, 50, 100]):
    metric = {}
    df[['sorted_score', 'sorted_candidate_collection']] = df.apply(sorting_collections_by_score, axis=1,
                                                                   result_type='expand')
    df['rank'] = df.apply(get_rank, axis=1, result_type='expand')
    df = cal_recall(df, top_k_list)
    df = cal_mrr(df, top_k_list)
    for top_k in tqdm(top_k_list):
        metric[f'Recall@{top_k}'] = df[f'Recall@{top_k}'].sum() / df.shape[0]
        metric[f'MRR@{top_k}'] = df[f'MRR@{top_k}'].sum() / df.shape[0]
    return metric


def sorting_collections_by_score(df):
    score = df['c_score_list']
    collection_list = df['c_collection_list']
    sorted_indices = np.argsort(score)[::-1]
    score, collection_list = np.array(score), np.array(collection_list)
    return score[sorted_indices], collection_list[sorted_indices]


def get_rank(df):
    y_label = df['q_collection_idx']
    sorted_candidate_collection = df['sorted_candidate_collection']
    for rank, candidate_collection in enumerate(sorted_candidate_collection):
        if y_label == candidate_collection:
            return rank + 1
    return 9999999


def cal_recall(df, top_k_list):
    for top_k in tqdm(top_k_list):
        df[f'Recall@{top_k}'] = 0
        df.loc[df['rank'] <= top_k, f'Recall@{top_k}'] = 1
    return df


def cal_mrr(df, top_k_list):
    for top_k in tqdm(top_k_list):
        df[f'MRR@{top_k}'] = df[f'Recall@{top_k}'] / df['rank']
    return df
