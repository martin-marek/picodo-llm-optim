import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P
from collections.abc import Mapping


def flatten_dict(d, prefix=None):
    if isinstance(d, Mapping):
        out = {}
        for k, v in d.items():
            nested_prefix = k if prefix is None else f'{prefix}.{k}'
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


def halflife_to_decay(t_token, n_batch):
    """
    notation:
    - t_token: halflife measured in number of tokens
    - t_steps: halflife measured in number of steps
    - n_batch: number of tokens per batch
    - d: decay coefficient
    """
    t_steps = t_token / n_batch # halflife (measured in number of steps)
    d = (1/2)**(1/t_steps)
    return d


def decay_to_halflife(d, n_batch):
    """
    notation:
    - t_token: halflife measured in number of tokens
    - t_steps: halflife measured in number of steps
    - n_batch: number of tokens per batch
    - d: decay coefficient
    """
    # note: d**t_steps = 1/2
    t_steps = jnp.log(1/2) / jnp.log(d)
    t_token = t_steps * n_batch
    return t_token


def flatten_model_dict(d):
    """
    flattens a fully-sharded model dictionary into a fully-sharded array
    - input: dictionary of nd-arrays
    - output: single array with shape [-1, device_count]
    """

    # we assume that the model is fully sharded across all devices
    shardings = P(None, 'data')
    n_devices = jax.device_count()

    # first, we reshape each leaf to [-1, device_count]
    flat = jax.tree.map(lambda x: jax.lax.with_sharding_constraint(x.reshape((-1, n_devices)), shardings), d)

    # second, we concatenate the leafs
    # importantly, no data has to move between devices here!
    flat = jnp.concatenate(jax.tree.leaves(flat)) # [-1, device_count]

    return flat

