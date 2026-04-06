import torch
from typing import Optional
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION, get_constant_schedule_with_warmup


class SchedulerWrapper:
    """Wrapper for ChainedScheduler to handle step() calls from accelerate."""
    def __init__(self, scheduler):
        self.scheduler = scheduler
    
    def step(self, *args, **kwargs):
        """Call step() without arguments (ChainedScheduler doesn't accept them)."""
        self.scheduler.step()
    
    def get_last_lr(self):
        """Get last learning rate."""
        return self.scheduler.get_last_lr()
    
    def __getattr__(self, name):
        """Delegate other attributes to wrapped scheduler."""
        return getattr(self.scheduler, name)


def get_lr_scheduler(
        name: Optional[str],
        optimizer: torch.optim.Optimizer,
        **kwargs,
):
    if name == "cosine":
        # Extract warmup_steps if present (not supported by CosineAnnealingLR directly)
        warmup_steps = kwargs.pop('warmup_steps', None)
        
        if 'total_iters' in kwargs:
            kwargs['T_max'] = kwargs.pop('total_iters')
        
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **kwargs
        )
        
        # If warmup is requested, chain warmup + cosine schedulers
        if warmup_steps is not None and warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup_steps
            )
            chained = torch.optim.lr_scheduler.ChainedScheduler(
                [warmup_scheduler, cosine_scheduler]
            )
            # Wrap to handle step() calls from accelerate
            return SchedulerWrapper(chained)
        
        return cosine_scheduler
    elif name == "cosine_with_restarts":
        if 'total_iters' in kwargs:
            kwargs['T_0'] = kwargs.pop('total_iters')
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, **kwargs
        )
    elif name == "step":

        return torch.optim.lr_scheduler.StepLR(
            optimizer, **kwargs
        )
    elif name == "constant":
        if 'factor' not in kwargs:
            kwargs['factor'] = 1.0

        return torch.optim.lr_scheduler.ConstantLR(optimizer, **kwargs)
    elif name == "linear":

        return torch.optim.lr_scheduler.LinearLR(
            optimizer, **kwargs
        )
    elif name == 'constant_with_warmup':
        # see if num_warmup_steps is in kwargs
        if 'num_warmup_steps' not in kwargs:
            print(f"WARNING: num_warmup_steps not in kwargs. Using default value of 1000")
            kwargs['num_warmup_steps'] = 1000
        del kwargs['total_iters']
        return get_constant_schedule_with_warmup(optimizer, **kwargs)
    else:
        # try to use a diffusers scheduler
        print(f"Trying to use diffusers scheduler {name}")
        try:
            name = SchedulerType(name)
            schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
            return schedule_func(optimizer, **kwargs)
        except Exception as e:
            print(e)
            pass
        raise ValueError(
            "Scheduler must be cosine, cosine_with_restarts, step, linear or constant"
        )
