import os
import sys
import logging

from PIL import Image
from torchvision import transforms

from dataclasses import dataclass, field
from typing import Optional
from setproctitle import setproctitle

import torch

from transformers import AutoImageProcessor, HfArgumentParser, TrainingArguments, set_seed, EfficientNetConfig, \
    Blip2Config, Trainer

from models import EfficientNetDualModel, BlipV2DualModel, AlphaBlipV2DualModel
from elements_datasets import CollectionDataset

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None

os.environ["WANDB_PROJECT"] = "[E2E] Match-Element-Retrieval"
os.environ["WANDB_LOG_MODEL"] = "false"  # "false"


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
    q_image_tensor = torch.stack([example["q_image_tensor"] for example in examples])  # 1
    c_image_tensor = torch.stack([example["c_image_tensor"] for example in examples])  # 1

    q_alpha_image_tensor = torch.stack([example["q_alpha_image_tensor"] for example in examples])  # 1
    c_alpha_image_tensor = torch.stack([example["c_alpha_image_tensor"] for example in examples])  # 1

    search_word = [example.get("search_word", "NOT_EXISTS") for example in examples]  #

    q_collection_idx = [example.get("q_collection_idx", "NOT_EXISTS") for example in examples]  # 1
    c_collection_idx = [example.get("c_collection_idx", "NOT_EXISTS") for example in examples]  # 2

    q_primary_element_key = [example.get("q_primary_element_key", "NOT_EXISTS") for example in examples]  # 1
    c_primary_element_key = [example.get("c_primary_element_key", "NOT_EXISTS") for example in examples]  # 1

    return {
        "q_image_tensor": q_image_tensor,
        "c_image_tensor": c_image_tensor,
        "q_alpha_image_tensor": q_alpha_image_tensor,
        "c_alpha_image_tensor": c_alpha_image_tensor,
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

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if model_args.backbone_name == 'BLIP-V2':
        config = Blip2Config.from_pretrained(model_args.model_name_or_path)
        config.logit_scale_init_value = model_args.logit_scale_init_value

        model = BlipV2DualModel.from_pretrained(model_args.model_name_or_path,
                                                cache_dir=model_args.cache_dir,
                                                config=config)
        alpha_processor = None
        processor = AutoImageProcessor.from_pretrained(model_args.model_name_or_path,
                                                       cache_dir=model_args.cache_dir)

    elif 'Alpha' in model_args.backbone_name:
        config = Blip2Config.from_pretrained(model_args.model_name_or_path)
        config.logit_scale_init_value = model_args.logit_scale_init_value

        model = AlphaBlipV2DualModel.from_pretrained(model_args.model_name_or_path,
                                                     cache_dir=model_args.cache_dir,
                                                     config=config)
        alpha_processor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(0.5, 0.26)
        ])
        processor = AutoImageProcessor.from_pretrained(model_args.model_name_or_path,
                                                       cache_dir=model_args.cache_dir)

    processor_fuc = {}
    processor_fuc['processor'] = processor
    processor_fuc['alpha_processor'] = alpha_processor

    set_seed(training_args.seed)

    train_dataset = CollectionDataset(data_args.train_dataset_path, data_args.element_image_path, processor_fuc)
    eval_dataset = CollectionDataset(data_args.eval_dataset_path, data_args.element_image_path, processor_fuc)

    if data_args.eval_ratio is not None:
        training_args.save_steps = training_args.eval_steps = round(
            (len(train_dataset) * training_args.num_train_epochs) / (
                    training_args.per_device_train_batch_size * data_args.num_device * training_args.gradient_accumulation_steps) * data_args.eval_ratio)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )
    train_result = trainer.train()
    trainer.save_model(training_args.output_dir)
    processor_fuc['processor'].save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    setproctitle('MIRIDIH-JSO')
    main()
