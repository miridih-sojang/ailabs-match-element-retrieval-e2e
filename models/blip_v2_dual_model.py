from typing import Any, Optional, Tuple, Union

import torch
from torch import nn

from dataclasses import dataclass

from transformers import Blip2Model
from transformers.utils import ModelOutput, logging

from loss import clip_loss, contrastive_loss

logger = logging.get_logger(__name__)


@dataclass
class ModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits_per_query: torch.FloatTensor = None
    logits_per_candidate: torch.FloatTensor = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(self[k] if k not in [] else getattr(self, k).to_tuple() for k in self.keys())


class BlipV2DualModel(Blip2Model):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.logit_scale = torch.nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
        self.reducenet = nn.Sequential(
            nn.Linear(1408, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
        )
        self.post_init()

    def forward(self,
                q_image_tensor: torch.FloatTensor = None,
                c_image_tensor: torch.FloatTensor = None,
                return_loss: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                search_word: Optional[str] = None,
                **kwargs) -> Union[Tuple, ModelOutput]:
        try:
            query_embedding_output = self.vision_model.embeddings(q_image_tensor)
            candidate_embedding_output = self.vision_model.embeddings(c_image_tensor)

            query_vision_outputs = self.vision_model.encoder(query_embedding_output)
            candidate_vision_outputs = self.vision_model.encoder(candidate_embedding_output)

            query_last_hidden_state = query_vision_outputs[0]
            query_last_hidden_state = self.vision_model.post_layernorm(query_last_hidden_state)
            query_pooled_output = query_last_hidden_state[:, 0, :]
            query_image_embeds = self.vision_model.post_layernorm(query_pooled_output)

            candidate_last_hidden_state = candidate_vision_outputs[0]
            candidate_last_hidden_state = self.vision_model.post_layernorm(candidate_last_hidden_state)
            candidate_pooled_output = candidate_last_hidden_state[:, 0, :]
            candidate_image_embeds = self.vision_model.post_layernorm(candidate_pooled_output)

            query_image_embeds = self.reducenet(query_image_embeds)
            candidate_image_embeds = self.reducenet(candidate_image_embeds)

            query_image_embeds = query_image_embeds / query_image_embeds.norm(p=2, dim=-1, keepdim=True)
            candidate_image_embeds = candidate_image_embeds / candidate_image_embeds.norm(p=2, dim=-1, keepdim=True)

            # cosine similarity as logits
            logit_scale = self.logit_scale.exp()
            logits_per_query = torch.matmul(query_image_embeds,
                                            candidate_image_embeds.t().to(query_image_embeds.device)) * logit_scale.to(
                query_image_embeds.device
            )
            logits_per_candidate = logits_per_query.t()

            loss = None
            if return_loss:
                loss = clip_loss(logits_per_query)

            if not return_dict:
                output = (logits_per_candidate, logits_per_query, query_image_embeds, candidate_image_embeds,
                          query_vision_outputs, candidate_vision_outputs)
                return ((loss,) + output) if loss is not None else output

            return ModelOutput(
                loss=loss,
                logits_per_query=logits_per_query,
                logits_per_candidate=logits_per_candidate
            )
        except Exception as e:
            torch.save(q_image_tensor, "./q_image_tensor.pt")
            torch.save(c_image_tensor, "./c_image_tensor.pt")

    def inference_image(self, image_tensor):
        with torch.inference_mode():
            embedding_output = self.vision_model.embeddings(image_tensor)
            vision_output = self.vision_model.encoder(embedding_output)
            last_hidden_state = vision_output[0]
            last_hidden_state = self.vision_model.post_layernorm(last_hidden_state)
            pooled_output = last_hidden_state[:, 0, :]
            image_embedding = self.vision_model.post_layernorm(pooled_output)
            image_embedding = image_embedding / image_embedding.norm(p=2, dim=-1, keepdim=True)
            return image_embedding

    def inference(self,
                  removed_pixel_values,
                  elements_pixel_values,
                  return_loss: Optional[bool] = False,
                  search_word: Optional[str] = None,
                  resourceKey: Optional[str] = None,
                  candidate_resourceKey: Optional[list] = None,
                  x_image_path: Optional[str] = None):

        element_image_embedding = self.inference_image(elements_pixel_values)
        background_images_embedding = self.inference_image(removed_pixel_values)
        _, similarity, _ = self.get_similarity(background_images_embedding, element_image_embedding)
        return similarity, (similarity)

    @staticmethod
    def get_similarity(query_image, candidate_image):
        similarity = torch.matmul(query_image, candidate_image.t())
        # similarity = torch.squeeze(similarity, dim=0)
        return None, similarity, ()
