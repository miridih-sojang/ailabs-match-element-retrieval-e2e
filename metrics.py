from tqdm import tqdm
import numpy as np
import pandas as pd


def evaluate(df, top_k_list=[1, 4, 5, 10, 20, 50, 100]):
    metric = {}
    df[['sorted_score', 'sorted_candidate_resourcekey']] = df.apply(sorting_resourcekey_by_score, axis=1,
                                                                    result_type='expand')
    df['rank'] = df.apply(get_rank, axis=1, result_type='expand')
    df = cal_recall(df, top_k_list)
    df = cal_mrr(df, top_k_list)
    df.to_csv('./metric.csv', index=False)
    for top_k in tqdm(top_k_list):
        metric[f'Recall@{top_k}'] = df[f'Recall@{top_k}'].sum() / df.shape[0]
        metric[f'MRR@{top_k}'] = df[f'MRR@{top_k}'].sum() / df.shape[0]
    return metric


def sorting_resourcekey_by_score(df):
    score = df['score']
    resourcekey_list = df['candidate_resourceKey']
    sorted_indices = np.argsort(score)[::-1]
    score, resourcekey_list = np.array(score), np.array(resourcekey_list)
    return score[sorted_indices], resourcekey_list[sorted_indices]


def get_rank(df):
    y_label = df['resourceKey']
    sorted_candidate_resourcekey = df['sorted_candidate_resourcekey']
    for rank, resourcekey in enumerate(sorted_candidate_resourcekey):
        if y_label == resourcekey:
            return rank + 1
    return 999999


def cal_recall(df, top_k_list):
    for top_k in tqdm(top_k_list):
        df[f'Recall@{top_k}'] = 0
        df.loc[df['rank'] <= top_k, f'Recall@{top_k}'] = 1
    return df


def cal_mrr(df, top_k_list):
    for top_k in tqdm(top_k_list):
        df[f'MRR@{top_k}'] = df[f'Recall@{top_k}'] / df['rank']
    return df


def compute_metrics(pred):
    test_vocabulary_path = '/data/match_element_retrieval/cache/search_dataset/test'
    test_dataset_df = pd.read_csv(f"{test_vocabulary_path}/csv/test_vocabulary.csv")
    test_dataset_df = test_dataset_df[~test_dataset_df.candidate_resourceKey.isna()]
    test_dataset_df['x_image_path'] = test_dataset_df['x_image_path'].apply(
        lambda x: f"{test_vocabulary_path}/images/{x}.png")
    test_dataset_df[['candidate_resourceKey', 'x_info', 'y_info']] = test_dataset_df[
        ['candidate_resourceKey', 'x_info', 'y_info']].map(lambda x: eval(x))
    test_dataset_df = test_dataset_df[test_dataset_df.candidate_resourceKey.str.len() < 400]
    test_dataset_df = test_dataset_df[
        ['search_word', 'template_idx', 'page_num', 'x_image_path', 'resourceKey', 'candidate_resourceKey']]
    test_dataset_df['score'] = pred.predictions.tolist()
    test_dataset_df['score'] = test_dataset_df['score'].apply(
        lambda x: np.array([instance_x for instance_x in x if instance_x != -10]))
    return evaluate(test_dataset_df)
