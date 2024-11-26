from torch import Tensor
import torch.distributed as dist
import torch
import torch.nn.functional as F


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


class CollectionContrastiveLoss:

    def __call__(self, query: Tensor, positive: Tensor, collection_idx: Tensor = None,
                 reduction: str = 'mean') -> Tensor:
        if collection_idx is None:
            raise ValueError("collection_idx is required")
        '''
        >>> collection_idx
            tensor([1, 2, 3, 2])
        >>> _repeat_collcection_idx
            tensor([[1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [3, 3, 3, 3],
                    [2, 2, 2, 2]])
        >>> target = (_repeat_collcection_idx == collection_idx.unsqueeze(0)).long()
            tensor([[1, 0, 0, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 1, 0, 1]])
        '''
        _repeat_collcection_idx = collection_idx.unsqueeze(-1).repeat((1, collection_idx.shape[0]))
        target = (_repeat_collcection_idx == collection_idx.unsqueeze(0)).long()
        target = target / torch.sum(target, dim=0)
        logits = torch.matmul(query, positive.transpose(0, 1))
        loss = F.cross_entropy(logits / self.temperature, target, reduction=reduction)
        return loss


class DistributedCollectionContrastiveLoss(CollectionContrastiveLoss):
    def __init__(self, n_target: int = 0, scale_loss: bool = True, temperature: float = 0.02):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__()
        self.word_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss
        self.temperature = temperature

    def __call__(self, x: Tensor, y: Tensor, collection_idx: Tensor = None, **kwargs):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)
        dist_collection_idx = self.gather_tensor(collection_idx)
        loss = super().__call__(dist_x, dist_y, dist_collection_idx, **kwargs)
        if self.scale_loss:
            loss = loss * self.word_size
        return loss

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)
