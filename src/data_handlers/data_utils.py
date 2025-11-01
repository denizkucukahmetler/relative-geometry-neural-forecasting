import torch
from torch.utils.data.dataloader import default_collate

def sequence_collate_fn(batch):
    """
    Custom collate function to handle batches containing tensors and custom objects
    like TensorNormalizer.

    Args:
        batch (list): A list of items where each item is a tuple (tensor, tensor, tensor, TensorNormalizer, tensor).
            Each item has the same tuple format as returned by DynamicalDataset.__getitem__

    Returns:
        tuple: A tuple (enc_inputs, dec_inputs, dec_targets, timesteps, scalers) where:
            - enc_inputs: torch.Tensor of shape (batch_size, enc_input_length, sample_dims)
            - dec_inputs: torch.Tensor of shape (batch_size, dec_input_length, sample_dims)
            - dec_targets: torch.Tensor of shape (batch_size, dec_target_length, sample_dims)
            - timesteps: torch.Tensor of shape (batch_size, dec_target_length)
            - scalers: np.array[TensorNormalizer] of length batch_size
    """
    enc_inputs = [item[0] for item in batch]  # All enc_input tensors
    dec_inputs = [item[1] for item in batch]  # All dec_input tensors
    dec_targets = [item[2] for item in batch]  # All dec_target tensors
    timesteps = [item[3] for item in batch]  # All timestep tensors
    scalers = [item[4] for item in batch]  # All scaler (TensorNormalizer) instances

    
    # Stack tensors as needed
    enc_inputs = torch.stack(enc_inputs)
    dec_inputs = torch.stack(dec_inputs)
    dec_targets = torch.stack(dec_targets)
    timesteps = torch.stack(timesteps)
    
    # Return a dictionary or tuple
    return enc_inputs, dec_inputs, dec_targets, timesteps, scalers
    
