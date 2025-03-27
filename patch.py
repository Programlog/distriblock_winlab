import torch
from speechbrain.utils.checkpoints import Checkpointer

def no_checkpoint_average(self, max_key=None, min_key=None):
    """Skip checkpoint averaging and use the pre-trained model directly"""
    super().on_evaluate_start()
    print("Skipping checkpoint averaging, using pre-trained model directly")
