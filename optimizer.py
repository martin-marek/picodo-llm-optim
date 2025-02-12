import jax
import jax.numpy as jnp
import optax
import operator as op
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
  elif optimizer_type == "adamw_ema":
    ema_decay = 0.5**(1/(c.adamw_ema_halflife * c.num_train_steps))
    base_optimizer = optax.inject_hyperparams(adamw_ema)(
      learning_rate_fn,
      ema_step_size=c.adamw_ema_step_size,
      ema_decay=ema_decay,
      b1=c.b1,
      b2=c.b2,
      eps=c.eps,
      weight_decay=c.weight_decay,
    )
  elif optimizer_type == "adamw_ema2":
    ema_decay1 = 0.5**(1/(c.adamw_ema_halflife * c.num_train_steps))
    ema_decay2 = 0.5**(1/(2*c.adamw_ema_halflife * c.num_train_steps)) # 2x the halflife of decay1
    base_optimizer = optax.inject_hyperparams(adamw_ema2)(
      learning_rate_fn,
      ema_step_size=c.adamw_ema_step_size,
      ema_decay1=ema_decay1,
      ema_decay2=ema_decay2,
      b1=c.b1,
      b2=c.b2,
      eps=c.eps,
      weight_decay=c.weight_decay,
    )
  else:
    raise ValueError(optimizer_type)

  return base_optimizer


class Ema2State(NamedTuple):
  ema1: base.Updates
  ema2: base.Updates
  step: jax.Array


def transform_add_ema_grad(
  step_size: float,
  decay: float,
) -> base.GradientTransformation:

  def init_fn(params):
    ema = otu.tree_zeros_like(params)
    step = jnp.array(0)
    return EmaState(ema, step)

  def update_fn(updates, state, params):
    ema = jax.tree.map(lambda e, p: decay*e + (1-decay)*p, state.ema, params)
    updates = jax.tree.map(lambda u, p, e: u + step_size*(e-p), updates, params, ema)

    return updates, optax.EmaState(ema, state.step+1)

  return base.GradientTransformation(init_fn, update_fn)


def transform_add_ema2_grad(
  step_size: float,
  decay1: float,
  decay2: float,
) -> base.GradientTransformation:

  def init_fn(params):
    ema1 = otu.tree_zeros_like(params)
    ema2 = otu.tree_zeros_like(params)
    step = jnp.array(0)
    return Ema2State(ema1, ema2, step)

  def update_fn(updates, state, params):
    # update ema
    ema1 = jax.tree.map(lambda e, p: decay1*e + (1-decay1)*p, state.ema1, params)
    ema2 = jax.tree.map(lambda e, p: decay2*e + (1-decay2)*p, state.ema2, params)

    # get projecttion
    def tree_project(v, s):
      """project 'v' onto 's'"""
      dot_nd = lambda x, y: jnp.dot(x.flatten(), y.flatten())
      tree_dot = lambda x, y: jax.tree.reduce(op.add, jax.tree.map(dot_nd, x, y))
      cp = tree_dot(v, s) / tree_dot(s, s) # projection scalar
      diff = jax.tree.map(lambda s, v: cp*s - v, s, v) # vector: v -> projection of v onto s
      return diff

    # ema2-ema1 projection
    v = jax.tree.map(op.sub, params, ema1) # the parameters vector we're projecting
    s = jax.tree.map(op.sub, ema2, ema1) # we're projecting onto the (e1-e2) line
    diff = tree_project(v, s)

    # take step toward projection
    updates = jax.tree.map(lambda u, d: u + step_size*d, updates, diff)

    return updates, Ema2State(ema1, ema2, state.step+1)

  return base.GradientTransformation(init_fn, update_fn)


def adamw_ema(
  learning_rate: base.ScalarOrSchedule,
  ema_step_size: float,
  ema_decay: float,
  b1: float = 0.9,
  b2: float = 0.999,
  eps: float = 1e-08,
  weight_decay: float = 1e-4,
  nesterov: bool = False,
) -> base.GradientTransformation:

  return combine.chain(
    transform.scale_by_adam(b1=b1, b2=b2, eps=eps, nesterov=nesterov),
    transform.add_decayed_weights(weight_decay),
    transform_add_ema_grad(ema_step_size, ema_decay), # <- this is the only modification
    transform.scale_by_learning_rate(learning_rate),
  )


def adamw_ema2(
  learning_rate: base.ScalarOrSchedule,
  ema_step_size: float,
  ema_decay1: float,
  ema_decay2: float,
  b1: float = 0.9,
  b2: float = 0.999,
  eps: float = 1e-08,
  weight_decay: float = 1e-4,
  nesterov: bool = False,
) -> base.GradientTransformation:

  return combine.chain(
    transform.scale_by_adam(b1=b1, b2=b2, eps=eps, nesterov=nesterov),
    transform.add_decayed_weights(weight_decay),
    transform_add_ema2_grad(ema_step_size, ema_decay1, ema_decay2), # <- this is the only modification
    transform.scale_by_learning_rate(learning_rate),
  )
