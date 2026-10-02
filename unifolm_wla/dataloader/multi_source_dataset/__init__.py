def create_training_dataloader(*args, **kwargs):
    """Load training dependencies only when constructing a dataloader.

    Robot clients also use action_mapping and se3_utils, without LeRobot,
    PyTorch, video decoders, or the training environment installed.
    """
    from .dataloader import create_training_dataloader as create

    return create(*args, **kwargs)

__all__ = ["create_training_dataloader"]
