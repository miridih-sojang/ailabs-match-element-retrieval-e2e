import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch

tqdm.pandas()


class CollectionDataset(torch.utils.data.Dataset):
    def __init__(self, path, image_path, image_processor):
        self.df = pd.read_csv(path)
        self.image_path = image_path
        self.image_processor = image_processor
        self.collection_list = self.df['collection_idx'].unique()

    def __getitem__(self, idx):
        query_df = self.df.iloc[idx]
        query_collection_idx, query_primary_element_key, query_file_path = query_df['collection_idx'], query_df[
            'primary_element_key'], query_df['file_path']
        candidate_df = self.df[(self.df.collection_idx == query_collection_idx) & (
                self.df.primary_element_key != query_primary_element_key)]
        candidate_df = candidate_df.sample(n=1).iloc[0]
        candidate_collection_idx, candidate_primary_element_key, candidate_file_path = candidate_df['collection_idx'], \
            candidate_df['primary_element_key'], candidate_df['file_path']

        query_image = self.read_image(query_file_path)
        candidate_image = self.read_image(candidate_file_path)
        query_image_tensor = self.image_processor(query_image, return_tensors='pt')['pixel_values'][0]
        candidate_image_tensor = self.image_processor(candidate_image, return_tensors='pt')['pixel_values'][0]
        # print(f'***************** IDX : {idx} ***************** {query_file_path} ||||| {candidate_file_path}',
        #       flush=True)
        # print(f'***************** IDX : {idx} ***************** {candidate_image_tensor.size()}', flush=True)

        # print(candidate_image_tensor)
        return {'q_image_tensor': query_image_tensor,
                'q_collection_idx': query_collection_idx,
                'q_primary_element_key': query_primary_element_key,
                'c_image_tensor': candidate_image_tensor,
                'c_primary_element_key': candidate_primary_element_key}

    def read_image(self, path):
        img = Image.open(f'{self.image_path}/{path}')
        img = img.convert('RGB')
        return img

    def __len__(self):
        return self.df.shape[0]


class ElementSameCollection(torch.utils.data.Dataset):
    def __init__(self, query_path, candidate_path, image_path):
        self.query_df = pd.read_csv(query_path)
        self.candidate_df = pd.read_csv(candidate_path)
        self.image_path = image_path

    def __getitem__(self, idx):
        candidate_df = self.candidate_df.iloc[idx]
        candidate_file_path, candidate_collection_idx = candidate_df['file_path'], candidate_df['collection_idx']
        return {'c_file_path': candidate_file_path, 'c_collection_idx': candidate_collection_idx}

    def __len__(self):
        return self.candidate_df.shape[0]


class ElementSameCollectionWithKeyword(torch.utils.data.Dataset):
    def __init__(self, path, image_path):
        self.df = pd.read_csv(path)
        self.image_path = image_path

    def __getitem__(self, idx):
        search_df = self.df.iloc[idx]

        search_df['q_primary_element_key'] = search_df.progress_apply(
            lambda x: f'{x["q_element_idx"]}-{x["q_element_type"]}', axis=1)
        search_df['c_primary_element_key'] = search_df.progress_apply(
            lambda x: f'{x["c_element_idx"]}-{x["c_element_type"]}', axis=1)
        search_df['a_primary_element_key'] = search_df.progress_apply(
            lambda x: f'{x["a_element_idx"]}-{x["a_element_type"]}', axis=1)

        return {'search_word': search_df['search_word'],
                'q_collection_idx': search_df['q_collection_idx'],
                'q_primary_element_key': search_df['q_primary_element_key'],
                'q_file_path': search_df['q_file_path'],
                'c_collection_idx': search_df['c_collection_idx'],
                'c_primary_element_key': search_df['c_primary_element_key'],
                'c_file_path': search_df['c_file_path'],
                'a_collection_idx': search_df['a_collection_idx'],
                'a_primary_element_key': search_df['a_primary_element_key'],
                'a_file_path': search_df['a_file_path'],
                }

    def __len__(self):
        return self.df.shape[0]
