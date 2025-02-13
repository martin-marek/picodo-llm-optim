import chex
import jax
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics
from collections.abc import Callable
from typing import Any, NamedTuple, Optional, Union


class MultiStepsState(NamedTuple):
  mini_step: chex.Array # Current mini-step counter. At an update, this either increases by 1 or is reset to 0.
  gradient_step: chex.Array # Gradient step counter. This only increases after enough mini-steps have been accumulated.
  inner_opt_state: Any # The state of the wrapped optimizer.
  acc_grads: Any # Accumulated gradients over multiple mini-steps.


class MultiSteps:
  """https://github.com/google-deepmind/optax/blob/b2e5820f71b43164cfee2eefe287a9692f8e3872/optax/transforms/_accumulation.py#L241#L428"""
  def __init__(
      self,
      opt: base.GradientTransformation,
      every_k_schedule: Union[int, Callable[[chex.Array], chex.Array]],
  ):
    self.inner_opt = opt
    self._every_k_schedule = lambda step: every_k_schedule if isinstance(every_k_schedule, int) else every_k_schedule

  def init(self, params: Any) -> MultiStepsState:
    init_state = MultiStepsState(
      mini_step=jnp.zeros([], dtype=jnp.int32),
      gradient_step=jnp.zeros([], dtype=jnp.int32),
      inner_opt_state=self.inner_opt.init(params),
      acc_grads=otu.tree_zeros_like(params),
    )
    return init_state

  def update(
      self,
      updates: base.Updates,
      state: MultiStepsState,
      params: Optional[base.Params] = None,
  ):
    k_steps = self._every_k_schedule(state.gradient_step)
    emit = state.mini_step == (k_steps - 1)

    # accumulate grads
    acc_grads = jax.tree.map(lambda grad, acc: acc + (grad - acc) / (state.mini_step + 1), updates, state.acc_grads)

    # if emit, do optimzier step
    # otherwise, return zero updates
    updates, inner_state = jax.lax.cond(emit,
      lambda: self.inner_opt.update(acc_grads, state.inner_opt_state, params=params),
      lambda: (otu.tree_zeros_like(updates), state.inner_opt_state),
    )

    # if emit, reset accumulated gradients
    acc_grads = jax.tree.map(lambda g: (1-emit)*g, acc_grads)

    # update state
    state = MultiStepsState(
        mini_step=(state.mini_step + 1) % k_steps,
        gradient_step=state.gradient_step + emit,
        inner_opt_state=inner_state,
        acc_grads=acc_grads,
    )

    return updates, state


  def gradient_transformation(self) -> base.GradientTransformation:
    return base.GradientTransformation(init=self.init, update=self.update)
