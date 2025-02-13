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


def welford_update(step, x, mean, m2):
  """https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm"""
  delta = x - mean
  mean += delta / step
  delta2 = x - mean
  m2 += delta * delta2
  return mean, m2


def tree_project(v, s):
  """project 'v' onto 's'"""
  dot_nd = lambda x, y: jnp.dot(x.flatten(), y.flatten())
  tree_dot = lambda x, y: jax.tree.reduce(op.add, jax.tree.map(dot_nd, x, y))
  cp = tree_dot(v, s) / tree_dot(s, s) # projection scalar
  diff = jax.tree.map(lambda s, v: cp*s - v, s, v) # vector: v -> projection of v onto s
  return diff
