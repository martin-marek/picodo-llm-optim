import jax
import jax.numpy as jnp
import optax
import operator as op
from optax import tree_utils as otu
from optax._src import base
from optax._src import combine
from optax._src import transform
from typing import NamedTuple, Optional, Literal
from omegaconf import OmegaConf
import utils
from multistep import MultiSteps


def get_learning_rate_schedule(c: OmegaConf) -> optax.Schedule:
  """Creates a learning rate schedule based on the config."""
  warmup_steps = int(c.warmup_frac * c.num_train_steps)
  cooldown_steps = int(c.cooldown_frac * c.num_train_steps)
  stable_steps = c.num_train_steps - warmup_steps - cooldown_steps

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
      optax.cosine_decay_schedule(init_value=c.peak_learning_rate, cooldown_steps=cooldown_steps)
    )
  elif c.cooldown_type == "linear":
    schedules.append(
      optax.linear_schedule(init_value=c.peak_learning_rate, end_value=0, transition_steps=cooldown_steps)
    )
  else:
    raise NotImplementedError(f"Unsupported decay type: {c.cooldown_type}")

  return optax.join_schedules(schedules, boundaries=[warmup_steps, warmup_steps+stable_steps])


def get_optimizer(c: OmegaConf) -> MultiSteps:
  learning_rate_fn = get_learning_rate_schedule(c)
  optimizer = optax.inject_hyperparams(adamw_ema)(
      learning_rate_fn,
      b1=c.b1,
      b2=c.b2,
      eps=c.eps,
      weight_decay=c.weight_decay,
      ema1_decay=0.5**(1/(c.ema1_halflife * c.num_train_steps)),
      ema2_decay=0.5**(1/(c.ema2_halflife * c.num_train_steps)),
      ema_update_type=c.ema_update_type,
      ema_step_size=c.ema_step_size,
    )
  optimizer = MultiSteps(optimizer, c.grad_accumulation_steps, c.grad_accumulation_bias)
  return optimizer


class EmaState(NamedTuple):
  step: jax.Array
  ema1: base.Params
  ema2: base.Params


def transform_add_ema(
  ema1_decay: float,
  ema2_decay: float,
  update_type: Optional[Literal['ema', 'ema_diff', 'ema_moment']] = None,
  step_size: float = 0.,
) -> base.GradientTransformation:

  def init_fn(params):
    ema1 = otu.tree_zeros_like(params)
    ema2 = otu.tree_zeros_like(params)
    step = jnp.array(0)
    return EmaState(step, ema1, ema2)

  def update_fn(updates, state, params):

    # update ema
    ema1 = jax.tree.map(lambda e, p: ema1_decay*e + (1-ema1_decay)*p, state.ema1, params)
    ema2 = jax.tree.map(lambda e, p: ema2_decay*e + (1-ema2_decay)*p, state.ema2, params)

    # step toward ema
    if update_type == 'ema':
      updates = jax.tree.map(lambda u, p, e: u + step_size*(e-p), updates, params, ema)

    # step toward ema diff projection
    if update_type == 'ema_diff':
      v = jax.tree.map(op.sub, params, state.ema1) # the parameters vector we're projecting
      s = jax.tree.map(op.sub,  state.ema2, state.ema1) # we're projecting onto the (e1-e2) line
      diff = utils.tree_project(v, s)
      updates = jax.tree.map(lambda u, d: u + step_size*d, updates, diff)

    return updates, EmaState(state.step+1, ema1, ema2)

  return base.GradientTransformation(init_fn, update_fn)


def adamw_ema(
  learning_rate: base.ScalarOrSchedule,
  b1: float = 0.9,
  b2: float = 0.999,
  eps: float = 1e-8,
  weight_decay: float = 1e-4,
  ema1_decay: float = 0.,
  ema2_decay: float = 0.,
  ema_update_type: Optional[Literal['ema', 'ema_diff', 'ema_moment']] = None,
  ema_step_size: float = 0.,
) -> base.GradientTransformation:
  """modified version of optax.adamw: https://github.com/google-deepmind/optax/blob/bff9977bec9aeccc63bf5fd3157d289a5b00694a/optax/_src/alias.py#L572"""

  return combine.chain(
    transform.scale_by_adam(b1=b1, b2=b2, eps=eps),
    transform.add_decayed_weights(weight_decay),
    transform_add_ema(ema1_decay, ema2_decay, ema_update_type, ema_step_size), # <- this is the only modification compared to adamw
    transform.scale_by_learning_rate(learning_rate),
  )
