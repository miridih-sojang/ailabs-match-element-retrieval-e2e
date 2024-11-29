from typing import Any, Optional, Tuple, Union

import torch
from torch import nn

from dataclasses import dataclass

from transformers import Blip2Model
from transformers.utils import ModelOutput, logging
from transformers.models.blip_2.modeling_blip_2 import Blip2VisionEmbeddings, Blip2VisionConfig

from loss import clip_loss, contrastive_loss

logger = logging.get_logger(__name__)


@dataclass
class ModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits_per_query: torch.FloatTensor = None
    logits_per_candidate: torch.FloatTensor = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(self[k] if k not in [] else getattr(self, k).to_tuple() for k in self.keys())


class AlphaBlip2VisionEmbeddings(Blip2VisionEmbeddings):
    def __init__(self, config: Blip2VisionConfig):
        super().__init__(config)
        self.alpha_patch_embedding = torch.nn.Conv2d(
            in_channels=1, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

    def forward(self, pixel_values: torch.FloatTensor, alpha_pixel_values: torch.FloatTensor = None,
                interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        alpha_patch_embeds = self.alpha_patch_embedding(
            alpha_pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds + alpha_patch_embeds
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        if interpolate_pos_encoding:
            position_embedding = self.interpolate_pos_encoding(embeddings, height, width)
        else:
            position_embedding = self.position_embedding
        embeddings = embeddings + position_embedding[:, : embeddings.size(1), :].to(target_dtype)
        return embeddings


class AlphaBlipV2DualModel(Blip2Model):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.logit_scale = torch.nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
        self.vision_model.embeddings = AlphaBlip2VisionEmbeddings(config.vision_config)
        # self.reducenet = torch.nn.Sequential(
        #     torch.nn.Linear(1408, 1024),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(1024, 512),
        #     torch.nn.Dropout(0.2),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(512, 256),
        #     torch.nn.ReLU(),
        # )
        self.reducenet = torch.nn.Sequential(
            torch.nn.Linear(1408, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
        )
        self.post_init()

    def forward(self,
                elements_pixel_values: torch.FloatTensor = None,
                removed_pixel_values: torch.FloatTensor = None,
                alpha_elements_pixel_values: torch.FloatTensor = None,
                alpha_removed_pixel_values: torch.FloatTensor = None):
        elements_embedding_output = self.vision_model.embeddings(elements_pixel_values, alpha_elements_pixel_values)
        removed_embedding_output = self.vision_model.embeddings(removed_pixel_values, alpha_removed_pixel_values)

        elements_vision_outputs = self.vision_model.encoder(elements_embedding_output)
        removed_vision_outputs = self.vision_model.encoder(removed_embedding_output)

        elements_last_hidden_state = elements_vision_outputs[0]
        elements_last_hidden_state = self.vision_model.post_layernorm(elements_last_hidden_state)
        elements_pooled_output = elements_last_hidden_state[:, 0, :]
        elements_image_embeds = self.vision_model.post_layernorm(elements_pooled_output)

        removed_last_hidden_state = removed_vision_outputs[0]
        removed_last_hidden_state = self.vision_model.post_layernorm(removed_last_hidden_state)
        removed_pooled_output = removed_last_hidden_state[:, 0, :]
        removed_image_embeds = self.vision_model.post_layernorm(removed_pooled_output)

        elements_image_embeds = self.reducenet(elements_image_embeds)
        removed_image_embeds = self.reducenet(removed_image_embeds)

        elements_image_embeds = elements_image_embeds / elements_image_embeds.norm(p=2, dim=-1, keepdim=True)
        removed_image_embeds = removed_image_embeds / removed_image_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_elements = torch.matmul(elements_image_embeds,
                                           removed_image_embeds.t().to(elements_image_embeds.device)) * logit_scale.to(
            elements_image_embeds.device
        )
        return logits_per_elements

    def inference_image(self, image_tensor, alpha_image_tensor):
        with torch.inference_mode():
            embedding_output = self.vision_model.embeddings(image_tensor, alpha_image_tensor)
            vision_output = self.vision_model.encoder(embedding_output)

            last_hidden_state = vision_output[0]
            last_hidden_state = self.vision_model.post_layernorm(last_hidden_state)
            pooled_output = last_hidden_state[:, 0, :]
            image_embedding = self.vision_model.post_layernorm(pooled_output)
            image_embedding = self.reducenet(image_embedding)
            image_embedding = image_embedding / image_embedding.norm(p=2, dim=-1, keepdim=True)
            return image_embedding
