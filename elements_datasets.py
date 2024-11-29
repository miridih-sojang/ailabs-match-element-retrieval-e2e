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


class CollectionCacheDataset(torch.utils.data.Dataset):
    def __init__(self, path, image_path, image_processor):
        self.df = pd.read_csv(path)
        self.image_path = image_path
        self.image_processor = image_processor

    def __getitem__(self, idx):
        df = self.df.iloc[idx]
        primary_element_key, collection_idx, file_path = df['primary_element_key'], df['collection_idx'], df[
            'file_path']
        img, alpha_img = self.read_image(file_path)
        img = self.image_processor['processor'](img, return_tensors='pt')['pixel_values'][0]
        if self.image_processor.get('alpha_processor', None) != None:
            alpha_img = self.image_processor['alpha_processor'](alpha_img)
        else:
            alpha_img = -100
        return img, alpha_img, primary_element_key, collection_idx

    def read_image(self, path):
        img = Image.open(f'{self.image_path}/{path}')
        img = img.convert('RGBA')
        alpha_img = img.split()[-1]
        img = img.convert('RGB')
        return img, alpha_img

    def __len__(self):
        return self.df.shape[0]
