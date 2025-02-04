import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu
from optax._src import base
from optax._src import combine
from optax._src import transform
from typing import NamedTuple
from omegaconf import OmegaConf


def get_optimizer(c: OmegaConf) -> optax.MultiSteps:
  """Get optimizer."""
  optimizer = _get_base_optimizer(c)
  if c.clip_by_global_norm:
    optimizer = optax.chain(optax.clip_by_global_norm(c.clip_by_global_norm), optimizer)
  optimizer = optax.MultiSteps(optimizer, c.grad_accumulation_steps)
  return optimizer


def get_learning_rate_schedule(c: OmegaConf) -> optax.Schedule:
  """Creates a learning rate schedule based on the config."""
  warmup_steps = int(c.warmup_frac * c.num_train_steps)
  decay_steps = int(c.decay_frac * c.num_train_steps)
  stable_steps = c.num_train_steps - warmup_steps - decay_steps

  # warmup
  schedules = [
    optax.linear_schedule(init_value=0, end_value=c.peak_learning_rate, transition_steps=warmup_steps)
  ]

  # stable
  schedules.append(
    optax.constant_schedule(value=c.peak_learning_rate)
  )

  # decay
  if c.decay_type == "cosine":
    decay_steps = c.get("decay_steps", c.num_train_steps - warmup_steps)
    schedules.append(
      optax.cosine_decay_schedule(init_value=c.peak_learning_rate, decay_steps=decay_steps)
    )
  elif c.decay_type == "linear":
    schedules.append(
      optax.linear_schedule(init_value=c.peak_learning_rate, end_value=0, transition_steps=decay_steps)
    )
  else:
    raise NotImplementedError(f"Unsupported decay type: {c.decay_type}")

  return optax.join_schedules(schedules, boundaries=[warmup_steps, warmup_steps+stable_steps])


def _get_base_optimizer(c: OmegaConf) -> optax.GradientTransformation:
  """Get base optimizer."""
  learning_rate_fn = get_learning_rate_schedule(c)
  optimizer_type = c.optimizer

  if optimizer_type == "adamw":
    base_optimizer = optax.inject_hyperparams(optax.adamw)(
      learning_rate_fn,
      b1=c.b1,
      b2=c.b2,
      eps=c.eps,
      weight_decay=c.weight_decay,
    )
  elif optimizer_type == "sgd":
    base_optimizer = optax.inject_hyperparams(optax.sgd)(
      learning_rate_fn,
      momentum=c.b1,
    )
  elif optimizer_type == "adamw_ewa":
    base_optimizer = optax.inject_hyperparams(adamw_ewa)(
      learning_rate_fn,
      ewa_step_size=c.ewa_step_size,
      ewa_decay=c.ewa_decay,
      ewa_zero_init=c.ewa_zero_init,
      b1=c.b1,
      b2=c.b2,
      eps=c.eps,
      weight_decay=c.weight_decay,
    )
  else:
    raise ValueError(optimizer_type)

  return base_optimizer


class EwaUpdateState(NamedTuple):
  ewa: base.Updates
  step: jax.Array


def transform_add_ewa_grad(
  step_size: float,
  decay: float,
  zero_init: bool,
) -> base.GradientTransformation:

  def init_fn(params):
    ewa = otu.tree_zeros_like(params)
    step = jnp.array(0)
    return EwaUpdateState(ewa, step)

  def update_fn(updates, state, params):
    ewa = jax.lax.cond(
      (~zero_init) & (state.step == 0),
      lambda: params,
      lambda: jax.tree.map(lambda e, p: decay*e + (1-decay)*p, state.ewa, params),
    )
    updates = jax.tree.map(lambda u, p, e: u + step_size*(e-p), updates, params, ewa)

    return updates, EwaUpdateState(ewa, state.step+1)

  return base.GradientTransformation(init_fn, update_fn)


def adamw_ewa(
  learning_rate: base.ScalarOrSchedule,
  ewa_step_size: float,
  ewa_decay: float,
  ewa_zero_init: bool,
  b1: float = 0.9,
  b2: float = 0.999,
  eps: float = 1e-08,
  weight_decay: float = 1e-4,
  nesterov: bool = False,
) -> base.GradientTransformation:

  return combine.chain(
    transform.scale_by_adam(b1=b1, b2=b2, eps=eps, nesterov=nesterov),
    transform.add_decayed_weights(weight_decay),
    transform_add_ewa_grad(ewa_step_size, ewa_decay, ewa_zero_init),
    transform.scale_by_learning_rate(learning_rate),
  )
