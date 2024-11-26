import os
import sys
import logging

from PIL import Image
import pandas as pd

from dataclasses import dataclass, field
from typing import Optional
from setproctitle import setproctitle

import torch

from transformers import AutoImageProcessor, HfArgumentParser, TrainingArguments, set_seed, EfficientNetConfig, \
    Blip2Config, Trainer

from models import EfficientNetDualModel, BlipV2DualModel
from trainer import MiridihTrainer
from metrics import compute_metrics
from elements_datasets import CollectionDataset, ElementSameCollection, ElementSameCollectionWithKeyword

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None

os.environ["WANDB_PROJECT"] = "[E2E] Match-Element-Retrieval"
os.environ["WANDB_LOG_MODEL"] = "end"  # "false"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    backbone_name: str = field(
        metadata={"help": "Use Backbone Name"},
    )
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    freeze_vision_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the vision model parameters or not."}
    )
    logit_scale_init_value: Optional[float] = field(
        default=2.6592,
        metadata={
            "help": (
                "logit_scale_init_value"
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the train_dataset."}
    )
    eval_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the eval_dataset."}
    )
    test_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the test_dataset."}
    )
    search_test_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the search_test_dataset."}
    )
    element_image_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the element image save directory."}
    )
    element_vector_cache_path: Optional[str] = field(
        default=None, metadata={"help": "element_vector_cache_path"}
    )
    image_column: Optional[str] = field(
        default="image_path",
        metadata={"help": "The name of the column in the datasets containing the full image file paths."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=32,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    eval_ratio: float = field(
        default=None,
        metadata={
            "help": (
                "Eval Step ratio"
            )
        },
    )
    num_device: Optional[int] = field(
        default=1,
        metadata={"help": "The number of GPU to use"},
    )

    def __post_init__(self):
        if self.train_dataset_path is None:
            raise ValueError("Need either a train_dataset_path.")


def collate_fn(examples):
    q_image_tensor = torch.stack([example["q_image_tensor"] for example in examples])
    c_image_tensor = torch.stack([example["c_image_tensor"] for example in examples])

    search_word = [example.get("search_word", "NOT_EXISTS") for example in examples]

    q_collection_idx = [example.get("q_collection_idx", "NOT_EXISTS") for example in examples]
    c_collection_idx = [example.get("c_collection_idx", "NOT_EXISTS") for example in examples]

    q_primary_element_key = [example.get("q_primary_element_key", "NOT_EXISTS") for example in examples]
    c_primary_element_key = [example.get("c_primary_element_key", "NOT_EXISTS") for example in examples]
    return {
        "q_image_tensor": q_image_tensor,
        "c_image_tensor": c_image_tensor,
        "search_word": search_word,
        "q_collection_idx": q_collection_idx,
        "c_collection_idx": c_collection_idx,
        "q_primary_element_key": q_primary_element_key,
        "c_primary_element_key": c_primary_element_key,
        "return_loss": True,
    }


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.report_to = ['wandb']
    # training_args.run_name = f'Qwen2-VL-72B-Instruct_20241023'

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if model_args.backbone_name == 'EfficientNet':
        config = EfficientNetConfig.from_pretrained(model_args.model_name_or_path)
        config.projection_dim = config.hidden_dim
        config.logit_scale_init_value = model_args.logit_scale_init_value

        model = EfficientNetDualModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )

        image_processor = AutoImageProcessor.from_pretrained(
            model_args.image_processor_name or model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )

    elif model_args.backbone_name == 'BLIP-V2':
        config = Blip2Config.from_pretrained(model_args.model_name_or_path)
        config.logit_scale_init_value = model_args.logit_scale_init_value

        model = BlipV2DualModel.from_pretrained(model_args.model_name_or_path,
                                                cache_dir=model_args.cache_dir,
                                                config=config)

        image_processor = AutoImageProcessor.from_pretrained(model_args.model_name_or_path,
                                                             cache_dir=model_args.cache_dir)

    set_seed(training_args.seed)

    train_dataset = CollectionDataset(data_args.train_dataset_path, data_args.element_image_path, image_processor)
    eval_dataset = CollectionDataset(data_args.eval_dataset_path, data_args.element_image_path, image_processor)
    test_dataset = ElementSameCollection(data_args.eval_dataset_path, data_args.test_dataset_path,
                                         data_args.element_image_path)
    search_test_dataset = ElementSameCollectionWithKeyword(data_args.search_test_dataset_path,
                                                           data_args.element_image_path)

    if data_args.eval_ratio is not None:
        training_args.save_steps = training_args.eval_steps = round(
            (len(train_dataset) * training_args.num_train_epochs) / (
                    training_args.per_device_train_batch_size * data_args.num_device * training_args.gradient_accumulation_steps) * data_args.eval_ratio)
    if model_args.backbone_name == 'EfficientNet':
        element_vector = torch.load(f'{data_args.element_vector_cache_path}/efficientnet.pth', weights_only=True)
    if model_args.backbone_name == 'BLIP-V2':
        element_vector = None  # torch.load(f'{data_args.element_vector_cache_path}/blip_v2.pth', weights_only=True)

    # trainer = MiridihTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset if training_args.do_train else None,
    #     eval_dataset=eval_dataset if training_args.do_eval else None,
    #     test_dataset=test_dataset,
    #     data_collator=collate_fn,
    #     rerank_compute_metrics=compute_metrics,
    #     element_vector=element_vector
    # )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        # test_dataset=test_dataset,
        data_collator=collate_fn,
        # rerank_compute_metrics=compute_metrics,
        # element_vector=element_vector
    )
    train_result = trainer.train()


if __name__ == "__main__":
    setproctitle('MIRIDIH-JSO')
    main()
