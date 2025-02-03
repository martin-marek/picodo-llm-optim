import jax
import numpy as np
import jax.numpy as jnp


def make_ds_loader(ds_path, seq_len, batch_size, offset_idx=0):
    """note: we assume that the dataset on the disk is already shuffled!"""

    # get num. tokens
    data = np.memmap(ds_path, dtype=np.uint16, mode='r')
    n_tokens = len(data) - offset_idx

    def get_batch(idx):

        # read dataset
        # using np.memmap for each batch to avoid memory leak
        data = np.memmap(ds_path, dtype=np.uint16, mode='r')

        # get batch
        start_idx = batch_size*seq_len*idx + seq_len*np.arange(batch_size)
        token_idx = start_idx[:, None] + np.arange(seq_len)[None, :] # [batch, sequence]
        batch = data[offset_idx + token_idx]

        return batch

    return get_batch, n_tokens


def get_in_out(batch: jax.Array, pad_id: int = 0):
  """Returns input, output, and weights for a batch of examples."""
  # Assumes input of the form <BOS> <IDs> <EOS> for eval.
  x = batch # [B, L]
  y = jnp.pad(x[:, 1:], ((0, 0), (0, 1)), constant_values=pad_id) # shift x by 1 along L axis
  weights = jnp.where(y != pad_id, 1, 0).astype(jnp.float32)
  return x, y, weights
