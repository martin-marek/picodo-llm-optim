import jax
import jax.numpy as jnp
import optax
import functools
import operator as op
from optax import tree_utils as otu
from optax._src import base
from optax._src import combine
from optax._src import transform
from typing import NamedTuple, Optional, Literal
from omegaconf import OmegaConf
import utils, multistep


def get_learning_rate_schedule(c: OmegaConf) -> optax.Schedule:
  """Creates a learning rate schedule based on the config."""
  optimizer_steps = c.num_train_steps * c.train_batch_size if c.single_step_training else c.num_train_steps // c.grad_accumulation_steps
  warmup_steps = int(c.warmup_frac * optimizer_steps)
  cooldown_steps = int(c.cooldown_frac * optimizer_steps)
  stable_steps = optimizer_steps - warmup_steps - cooldown_steps

  # warmup
  schedules = [
    optax.linear_schedule(init_value=0, end_value=c.peak_learning_rate, transition_steps=warmup_steps)
  ]

  # stable
  schedules.append(
    optax.constant_schedule(value=c.peak_learning_rate)
  )

  # decay
  if c.cooldown_type == "cosine":
    schedules.append(
      optax.cosine_decay_schedule(init_value=c.peak_learning_rate, decay_steps=cooldown_steps)
    )
  elif c.cooldown_type == "linear":
    schedules.append(
      optax.linear_schedule(init_value=c.peak_learning_rate, end_value=0, transition_steps=cooldown_steps)
    )
  else:
    raise NotImplementedError(f"Unsupported decay type: {c.cooldown_type}")

  return optax.join_schedules(schedules, boundaries=[warmup_steps, warmup_steps+stable_steps])


def get_optimizer(c: OmegaConf, tokens_per_train_batch: int):
  adam_t1 = c.adam_t1 # halflife measured in num. tokens
  adam_t2 = c.adam_t1 * c.adam_t2_t1_ratio
  adam_b1 = utils.halflife_to_decay(adam_t1, tokens_per_train_batch)
  adam_b2 = utils.halflife_to_decay(adam_t2, tokens_per_train_batch)
  ema1_decay = utils.halflife_to_decay(c.ema1_halflife, tokens_per_train_batch)
  ema2_decay = utils.halflife_to_decay(c.ema2_halflife, tokens_per_train_batch)
  learning_rate_fn = get_learning_rate_schedule(c)
  multistep_wrapper = multistep.SingleSteps if c.grad_accumulation_steps==1 else multistep.MultiSteps
  if c.optimizer == "adamw2":
    optimizer = optax.inject_hyperparams(
      lambda learning_rate, b1, b2, eps, weight_decay, steps, bias: 
        multistep_wrapper(adamw2(learning_rate, b1, b2, eps, weight_decay), steps, bias)
      )(learning_rate=learning_rate_fn,
        b1=adam_b1,
        b2=adam_b2,
        eps=c.eps,
        weight_decay=c.weight_decay,
        steps=c.grad_accumulation_steps,
        bias=c.grad_accumulation_bias,
        # ema1_decay=ema1_decay,
        # ema2_decay=ema2_decay,
        # ema_update_type=c.ema_update_type,
        # ema_step_size=c.ema_step_size,
      )
  else:
    raise ValueError(c.optimizer)
  return optimizer


class ScaleByAdamW2State(NamedTuple):
  step: jax.Array
  m1: base.Params # ema of g
  m2: base.Params # ema of g**2


def adamw2(
    learning_rate: float,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 1e-4,
  ) -> base.GradientTransformation:

  def init_fn(params):
    step = jnp.zeros([], jnp.int32)
    m1 = otu.tree_zeros_like(params)
    m2 = otu.tree_zeros_like(params)
    return ScaleByAdamW2State(step, m1, m2)

  def update_fn(updates, state, params, **kwargs):

    # scale by adam
    m1 = otu.tree_update_moment(updates, state.m1, b1, 1)
    m2 = otu.tree_update_moment_per_elem_norm(updates, state.m2, b2, 2)
    m1_hat = otu.tree_bias_correction(m1, b1, state.step+1)
    m2_hat = otu.tree_bias_correction(m2, b2, state.step+1)
    updates = jax.tree.map(lambda m, v: m / (jnp.sqrt(v) + eps), m1_hat, m2_hat)

    # add weight decay
    updates = jax.tree.map(lambda g, p: g + weight_decay * p, updates, params)

    # scale by lr
    updates = jax.tree.map(lambda g: -learning_rate * g, updates)

    return updates, ScaleByAdamW2State(state.step+1, m1, m2)

  return base.GradientTransformation(init_fn, update_fn)

