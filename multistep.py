import chex
import jax
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics
from collections.abc import Callable
from typing import Any, NamedTuple, Optional, Union
import utils


class MultiStepsState(NamedTuple):
    mini_step: chex.Array # Current mini-step counter. At an update, this either increases by 1 or is reset to 0.
    gradient_step: chex.Array # Gradient step counter. This only increases after enough mini-steps have been accumulated.
    inner_opt_state: Any # The state of the wrapped optimizer.
    grad_stats: Any # Accumulated gradients over multiple mini-steps.


class SingleStepsState(NamedTuple):
    inner_opt_state: Any # The state of the wrapped optimizer.


class MultiSteps:
    """https://github.com/google-deepmind/optax/blob/b2e5820f71b43164cfee2eefe287a9692f8e3872/optax/transforms/_accumulation.py#L241#L428"""
    def __init__(
        self,
        opt: base.GradientTransformation,
        steps: int,
    ):
        self.inner_opt = opt
        self.steps = steps

    def init(self, params: Any) -> MultiStepsState:
        init_state = MultiStepsState(
            mini_step=jnp.zeros([], dtype=jnp.int32),
            gradient_step=jnp.zeros([], dtype=jnp.int32),
            inner_opt_state=self.inner_opt.init(params),
            grad_stats = jax.tree.map(lambda x: (0.*x, 0.*x), params) # (mean, m2)
        )
        return init_state

    def update(
        self,
        updates: base.Updates,
        state: MultiStepsState,
        params: Optional[base.Params] = None,
    ):
        emit = state.mini_step >= (self.steps - 1)

        # accumulate grads
        grad_stats = jax.tree.map(lambda g, stats: utils.welford_update(state.mini_step+1, g, *stats), updates, state.grad_stats)

        # get grad estimate
        grad_mean = jax.tree.map(lambda _, stats: stats[0], updates, grad_stats)
        grad_std = jax.tree.map(lambda _, stats: jnp.sqrt(stats[1]/(state.mini_step+1)), updates, grad_stats)

        # if emit, do optimzier step
        # otherwise, return zero updates
        updates, inner_state = jax.lax.cond(emit,
            lambda: self.inner_opt.update(grad_mean, state.inner_opt_state, params=params, grad_std=grad_std),
            lambda: (otu.tree_zeros_like(updates), state.inner_opt_state),
        )

        # if emit, reset accumulated gradients
        grad_stats = jax.tree.map(lambda g: (1-emit)*g, grad_stats)

        # update state
        state = MultiStepsState(
            mini_step=(state.mini_step + 1) * emit,
            gradient_step=state.gradient_step + emit,
            inner_opt_state=inner_state,
            grad_stats=grad_stats,
        )

        return updates, state

    def gradient_transformation(self) -> base.GradientTransformation:
        return base.GradientTransformation(init=self.init, update=self.update)


class SingleSteps:
    """https://github.com/google-deepmind/optax/blob/b2e5820f71b43164cfee2eefe287a9692f8e3872/optax/transforms/_accumulation.py#L241#L428"""
    def __init__(self, opt: base.GradientTransformation, *args, **kwargs):
        self.inner_opt = opt

    def init(self, params: Any) -> MultiStepsState:
        init_state = SingleStepsState(self.inner_opt.init(params))
        return init_state

    def update(
        self,
        updates: base.Updates,
        state: MultiStepsState,
        params: Optional[base.Params] = None,
        **kwargs,
    ):
        updates, inner_state = self.inner_opt.update(updates, state.inner_opt_state, params=params, **kwargs)
        state = SingleStepsState(inner_state)
        return updates, state

    def gradient_transformation(self) -> base.GradientTransformation:
        return base.GradientTransformation(init=self.init, update=self.update)

