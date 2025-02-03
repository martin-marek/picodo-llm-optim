import jax
import jax.numpy as jnp
from flax import nnx
from collections.abc import Mapping


def flatten_dict(d, prefix=None):
  if isinstance(d, Mapping):
    out = {}
    for k, v in d.items():
      nested_prefix = k if prefix is None else f'{prefix}/{k}'
      out |= flatten_dict(v, nested_prefix)
    return out
  else:
    return {prefix: d}


def get_num_model_params(model: nnx.Module):
  graphdef, params = nnx.split(model, nnx.Param)
  n_params = jax.tree.reduce(lambda x, y: x + jnp.size(y), params, 0)
  return n_params
