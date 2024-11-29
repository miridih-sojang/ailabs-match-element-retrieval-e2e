import os
import argparse
import numpy as np
from numpy.linalg import norm
import pandas as pd
from setproctitle import setproctitle
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from transformers.models.blip_2.modeling_blip_2 import Blip2Config
from models import BlipV2DualModel, AlphaBlipV2DualModel
from elements_datasets import CollectionCacheDataset
from utils import read_yaml, write_yaml


def get_args():
    parser = argparse.ArgumentParser(description='MIRIDIH Matching Element Retrieval E2E Evaluation')
    parser.add_argument('--dataset_config_path', type=str, help='Dataset Config Path')
    parser.add_argument('--model_config_path', type=str, help='Model Config Path')
    parser.add_argument('--experiment_config_path', type=str, help='Experiment Config Path')
    args = parser.parse_args()
    return args


def cosine_similarity(a, b):
    return (np.dot(a, b) / (norm(a) * norm(b)) + 1) / 2


def main():
    args = get_args()
    dataset_config, model_config, experiment_config = (read_yaml(args.dataset_config_path),
                                                       read_yaml(args.model_config_path),
                                                       read_yaml(args.experiment_config_path))

    os.makedirs(f'{experiment_config["save_path"]}/{experiment_config["exp_name"]}', exist_ok=True)
    write_yaml(dataset_config, f'{experiment_config["save_path"]}/{experiment_config["exp_name"]}',
               'dataset_config.yaml')
    write_yaml(model_config, f'{experiment_config["save_path"]}/{experiment_config["exp_name"]}',
               'model_config.yaml')
    write_yaml(experiment_config, f'{experiment_config["save_path"]}/{experiment_config["exp_name"]}',
               'experiment_config.yaml')

    processor_fuc = {}
    if model_config["name"] == 'BLIP-V2':
        processor = AutoImageProcessor.from_pretrained(model_config['trained_path'])
        config = Blip2Config.from_pretrained(model_config['trained_path'])
        config.logit_scale_init_value = model_config['logit_scale_init_value']
        model = BlipV2DualModel.from_pretrained(model_config['trained_path'], config=config)
        alpha_processor = None
    elif 'Alpha' in model_config["name"]:
        processor = AutoImageProcessor.from_pretrained(model_config['trained_path'])
        config = Blip2Config.from_pretrained(model_config['trained_path'])
        config.logit_scale_init_value = model_config['logit_scale_init_value']
        model = AlphaBlipV2DualModel.from_pretrained(model_config['trained_path'], config=config)
        alpha_processor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(0.5, 0.26)
        ])

    processor_fuc['processor'] = processor
    processor_fuc['alpha_processor'] = alpha_processor

    model.cuda()
    model.eval()

    if experiment_config['scenario_1'] == 'Use' or experiment_config['scenario_2'] == 'Use':
        cache_dataset = CollectionCacheDataset(dataset_config['path']['total_dataset'],
                                               dataset_config['path']['image_path'],
                                               processor_fuc)
        cache_dataloader = DataLoader(cache_dataset, batch_size=256, num_workers=8, shuffle=False, drop_last=False)
        vector_cache = {}
        for batch in tqdm(cache_dataloader):
            img, alpha_img, primary_element_key, collection_idx = batch
            img = img.cuda()
            if processor_fuc['alpha_processor'] != None:
                alpha_img = alpha_img.cuda()
            if processor_fuc['alpha_processor'] != None:
                embeddings = model.inference_image(img, alpha_img).cpu().detach().numpy()
            else:
                embeddings = model.inference_image(img).cpu().detach().numpy()
            for embedding, p_e_k in zip(embeddings, primary_element_key):
                if vector_cache.get(p_e_k, None) is None:
                    vector_cache[p_e_k] = embedding
    if experiment_config['scenario_1'] == 'Use':
        query_df = pd.read_csv(dataset_config['path']['test_dataset'])
        scenario_1_predictions = []
        for q_primary_element_key, q_collection_idx in tqdm(query_df[['primary_element_key', 'collection_idx']].values):
            score_list = []
            c_key_list = []
            c_collection_list = []
            for c_primary_element_key, c_collection_idx in query_df[['primary_element_key', 'collection_idx']].values:
                if q_primary_element_key == c_primary_element_key:
                    continue
                else:
                    score_list.append(
                        cosine_similarity(vector_cache[q_primary_element_key], vector_cache[c_primary_element_key]))

                    c_key_list.append(c_primary_element_key)
                    c_collection_list.append(c_collection_idx)
            scenario_1_predictions.append(
                [q_primary_element_key, q_collection_idx, c_key_list, score_list, c_collection_list])
        scenario_1_prediction_df = pd.DataFrame(scenario_1_predictions,
                                                columns=['q_primary_element_key', 'q_collection_idx', 'c_key_list',
                                                         'c_score_list', 'c_collection_list'])
        scenario_1_prediction_df.to_csv(
            f'{experiment_config["save_path"]}/{experiment_config["exp_name"]}/scenario_1_prediction.csv', index=False)

    if experiment_config['scenario_2'] == 'Use':
        search_test_df = pd.read_csv(dataset_config['path']['search_test_dataset'])
        search_test_df['c_collection_idx'] = search_test_df['c_collection_idx'].apply(lambda x: eval(x))
        search_test_df['c_element_idx'] = search_test_df['c_element_idx'].apply(lambda x: eval(x))
        search_test_df['c_element_type'] = search_test_df['c_element_type'].apply(lambda x: eval(x))
        scenario_2_predictions = []
        for search_word, q_collection_idx, q_element_idx, q_element_type, c_collection_idx, c_element_idx, c_element_type in tqdm(
                search_test_df[
                    ['search_word', 'q_collection_idx', 'q_element_idx', 'q_element_type', 'c_collection_idx',
                     'c_element_idx', 'c_element_type']].values):
            q_primary_element_key = f'{q_element_idx}-{q_element_type}'
            score_list = []
            c_key_list = []
            c_collection_list = []
            for ins_c_collection_idx, ins_c_element_idx, ins_c_element_type in zip(c_collection_idx, c_element_idx,
                                                                                   c_element_type):
                c_primary_element_key = f'{ins_c_element_idx}-{ins_c_element_type}'
                if q_primary_element_key == c_primary_element_key:
                    continue
                else:
                    score_list.append(
                        cosine_similarity(vector_cache[q_primary_element_key], vector_cache[c_primary_element_key]))
                    c_key_list.append(c_primary_element_key)
                    c_collection_list.append(ins_c_collection_idx)
            scenario_2_predictions.append(
                [search_word, q_primary_element_key, q_collection_idx, c_key_list, score_list, c_collection_list])
        scenario_2_prediction_df = pd.DataFrame(scenario_2_predictions,
                                                columns=['search_word', 'q_primary_element_key', 'q_collection_idx',
                                                         'c_key_list', 'c_score_list', 'c_collection_list'])
        scenario_2_prediction_df.to_csv(
            f'{experiment_config["save_path"]}/{experiment_config["exp_name"]}/scenario_2_prediction.csv', index=False)


if __name__ == "__main__":
    setproctitle('MIRIDIH-JSO-Inference')
    main()
