import math
import torch


def get_pruning_schedule(target, num_iter):
    p = math.pow(target, 1 / num_iter)
    schedule = [p ** i for i in range(1, num_iter)] + [target]
    return schedule


def gumbel_sigmoid(logits: torch.Tensor, temperature: float = 1.0, training: bool = True, use_gumbel: bool = True) -> torch.Tensor:
    """Apply Gumbel-Sigmoid: sigmoid((logits + gumbel_noise) / temperature).

    Args:
        logits: Input logits to apply Gumbel-Sigmoid to.
        temperature: Temperature parameter. Lower values make the output more discrete.
        use_gumbel: If False, disables Gumbel noise and uses normal sigmoid.
    Returns:
        The Gumbel-Sigmoid output.
    """
    if use_gumbel and training:
        # Sample Gumbel noise: g = -log(-log(u)) where u ~ Uniform(0,1)
        u = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(u + 1e-20) + 1e-20)
        return torch.sigmoid(logits + (gumbel_noise * temperature))
    else:
        # During eval, just apply temperature-scaled sigmoid without noise
        return torch.sigmoid(logits)


def get_gumbel_temperature(step: int, total_steps: int, temp_start: float, temp_end: float, anneal_type: str = 'linear') -> float:
    """Compute current Gumbel temperature based on annealing schedule.
    
    Args:
        step: Current training step.
        total_steps: Total number of training steps.
        temp_start: Starting temperature.
        temp_end: Ending temperature.
        anneal_type: Type of annealing schedule ('linear', 'exponential', or 'constant').
    
    Returns:
        Current temperature value.
    """
    if anneal_type == 'constant':
        return temp_start
    elif anneal_type == 'linear':
        progress = step / total_steps
        return temp_start + (temp_end - temp_start) * progress
    elif anneal_type == 'exponential':
        progress = step / total_steps
        return temp_start * (temp_end / temp_start) ** progress
    return temp_start
