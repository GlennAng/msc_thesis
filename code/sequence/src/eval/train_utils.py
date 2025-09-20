import torch

from ..models.recommender import Recommender


def load_optimizer(
    recommender: Recommender,
    lr: float,
    lr_scheduler: str,
    weight_decay: float,
    n_batches_total: int,
    n_warmup_steps: int,
) -> tuple:
    if sum(p.numel() for p in recommender.parameters()) == 0:
        return None, None
    optimizer = torch.optim.Adam(recommender.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_scheduler == "constant":
        return optimizer, None
    elif lr_scheduler == "linear_decay":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, n_total_steps=n_batches_total, n_warmup_steps=n_warmup_steps
        )
        return optimizer, scheduler
    return optimizer, None


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, n_total_steps: int, n_warmup_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(current_step: int):
        if current_step < n_warmup_steps:
            return float(current_step) / float(n_warmup_steps)
        else:
            return max(
                0.0, float(n_total_steps - current_step) / float(n_total_steps - n_warmup_steps)
            )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)


def process_batch(batch: dict, recommender: Recommender) -> tuple:
    assert recommender.training
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(recommender.get_device())
    return recommender(batch)


def compute_info_nce_loss(
    dot_products: torch.Tensor, 
    mask_negrated: torch.Tensor, 
    temperature: float
) -> torch.Tensor:
    logits = dot_products / temperature
    mask_shape = mask_negrated.shape[1]
    valid_mask = torch.ones_like(logits, dtype=torch.bool)
    valid_mask[:, 1:1+mask_shape] = mask_negrated
    masked_logits = logits.masked_fill(~valid_mask, float('-inf'))
    positive_logits = masked_logits[:, 0]    
    log_sum_exp = torch.logsumexp(masked_logits, dim=1)
    loss = -positive_logits + log_sum_exp    
    return loss.mean()