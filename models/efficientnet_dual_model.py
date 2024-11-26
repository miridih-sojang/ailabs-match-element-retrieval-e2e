from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from dataclasses import dataclass

from transformers import EfficientNetPreTrainedModel
from transformers.models.efficientnet.modeling_efficientnet import EfficientNetEmbeddings, EfficientNetConfig, \
    EfficientNetEncoder
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import ModelOutput, logging

logger = logging.get_logger(__name__)


@dataclass
class EffOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits_per_removed: torch.FloatTensor = None
    logits_per_elements: torch.FloatTensor = None
    elements_embeds: torch.FloatTensor = None
    removed_embeds: torch.FloatTensor = None
    elements_model_output: BaseModelOutputWithPooling = None
    removed_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["elements_model_output", "removed_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


@dataclass
class CosOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    similarity: Optional[torch.FloatTensor] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["similarity"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


class EfficientNetDualModel(EfficientNetPreTrainedModel):
    def __init__(self, config: EfficientNetConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = EfficientNetEmbeddings(config)
        self.encoder = EfficientNetEncoder(config)

        # Final pooling layer
        if config.pooling_type == "mean":
            self.pooler = nn.AvgPool2d(config.hidden_dim, ceil_mode=True)
        elif config.pooling_type == "max":
            self.pooler = nn.MaxPool2d(config.hidden_dim, ceil_mode=True)
        else:
            raise ValueError(f"config.pooling must be one of ['mean', 'max'] got {config.pooling}")

        self.vision_embed_dim = config.hidden_dim
        self.projection_dim = config.projection_dim

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            elements_pixel_values: torch.FloatTensor = None,
            removed_pixel_values: torch.FloatTensor = None,
            return_loss: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            search_word: Optional[str] = None,
            resourceKey: Optional[str] = None,
            candidate_resourceKey: Optional[str] = None
    ) -> Union[Tuple, EffOutput]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if elements_pixel_values is None:
            raise ValueError("You have to specify elements_pixel_values")
        if removed_pixel_values is None:
            raise ValueError("You have to specify removed_pixel_values")

        elements_embedding_output = self.embeddings(elements_pixel_values)
        removed_embedding_output = self.embeddings(removed_pixel_values)

        elements_vision_outputs = self.encoder(
            elements_embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        removed_vision_outputs = self.encoder(
            removed_embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Apply pooling
        elements_last_hidden_state = elements_vision_outputs[0]
        elements_pooled_output = self.pooler(elements_last_hidden_state)
        # Reshape (batch_size, 1792, 1 , 1) -> (batch_size, 1792)
        elements_pooled_output = elements_pooled_output.reshape(elements_pooled_output.shape[:2])

        elements_image_embeds = self.visual_projection(elements_pooled_output)

        # Apply pooling
        removed_last_hidden_state = removed_vision_outputs[0]
        removed_pooled_output = self.pooler(removed_last_hidden_state)
        # Reshape (batch_size, 1792, 1 , 1) -> (batch_size, 1792)
        removed_pooled_output = removed_pooled_output.reshape(removed_pooled_output.shape[:2])

        removed_image_embeds = self.visual_projection(removed_pooled_output)

        # normalized features
        elements_image_embeds = elements_image_embeds / elements_image_embeds.norm(p=2, dim=-1, keepdim=True)
        removed_image_embeds = removed_image_embeds / removed_image_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_elements = torch.matmul(elements_image_embeds,
                                           removed_image_embeds.t().to(elements_image_embeds.device)) * logit_scale.to(
            elements_image_embeds.device
        )
        logits_per_removed = logits_per_elements.t()

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_elements)

        if not return_dict:
            output = (logits_per_removed, logits_per_elements, elements_image_embeds, removed_image_embeds,
                      elements_vision_outputs, removed_vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return EffOutput(
            loss=loss,
            logits_per_removed=logits_per_removed,
            logits_per_elements=logits_per_elements,
            elements_embeds=elements_image_embeds,
            removed_embeds=removed_image_embeds,
            elements_model_output=elements_vision_outputs,
            removed_model_output=removed_vision_outputs,
        )

    def inference_image(self, image_tensor):
        with torch.inference_mode():
            embedding_output = self.embeddings(image_tensor)
            vision_output = self.encoder(embedding_output)
            last_hidden_state = vision_output[0]
            pooled_output = self.pooler(last_hidden_state).reshape(last_hidden_state.shape[0], -1)
            image_embedding = self.visual_projection(pooled_output)
            image_embedding = image_embedding / image_embedding.norm(p=2, dim=-1, keepdim=True)
            return image_embedding

    def inference(self,
                  removed_pixel_values,
                  elements_pixel_values,
                  return_loss: Optional[bool] = False,
                  search_word: Optional[str] = None,
                  resourceKey: Optional[str] = None,
                  candidate_resourceKey: Optional[list] = None):

        element_image_embedding = self.inference_image(elements_pixel_values)
        background_images_embedding = self.inference_image(removed_pixel_values)
        similarity = self.get_similarity(background_images_embedding, element_image_embedding)
        return similarity, (similarity)

    @staticmethod
    def get_similarity(query_image, candidate_image):
        similarity = torch.matmul(query_image, candidate_image.t())
        # similarity = torch.squeeze(similarity, dim=0)
        return similarity
