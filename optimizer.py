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
    optimizer_steps = c.num_train_steps * c.train_batch_size if c.single_step_training else c.num_train_steps
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


def get_optimizer(c: OmegaConf, tokens_per_microbatch: int):
    learning_rate_fn = get_learning_rate_schedule(c)
    # steps = c.grad_accumulation_steps
    steps = lambda x: x/3
    multistep_wrapper = multistep.SingleSteps if steps==1 else multistep.MultiSteps
    assert c.optimizer == "adamw2"
    optimizer = optax.inject_hyperparams(
        lambda learning_rate, t1, r, weight_decay, steps: 
            multistep_wrapper(
                adamw2(
                    learning_rate=learning_rate,
                    tokens_per_microbatch=tokens_per_microbatch,
                    t1=t1,
                    r=r,
                    eps=c.eps,
                    weight_decay=weight_decay,
                ),
                steps
            )
        )(learning_rate=learning_rate_fn,
          t1=c.adam_t1,
          r=c.adam_t2_t1_ratio,
          weight_decay=c.weight_decay,
          steps=steps,
        )
    return optimizer


class ScaleByAdamW2State(NamedTuple):
    step: jax.Array
    m1: base.Params # ema of g
    m2: base.Params # ema of g**2


def adamw2(
        learning_rate: float,
        tokens_per_microbatch: int,
        t1: float = 2_000_000, # β1 decay half-life in num. tokens
        r: float = 0.999, # ratio of β2/β1 decay half-life
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
    ) -> base.GradientTransformation:

    # convert half-lives to decay values
    t2 = t1 * r # β2 decay half-life
    b1 = utils.halflife_to_decay(t1, tokens_per_microbatch) # β1 decay coefficient
    b2 = utils.halflife_to_decay(t2, tokens_per_microbatch) # β2 decay coefficient
    # ema1_decay = utils.halflife_to_decay(c.ema1_halflife, tokens_per_microbatch)
    # ema2_decay = utils.halflife_to_decay(c.ema2_halflife, tokens_per_microbatch)

    def init_fn(params):
        step = jnp.zeros([], jnp.int32)
        m1 = otu.tree_zeros_like(params)
        m2 = otu.tree_zeros_like(params)
        return ScaleByAdamW2State(step, m1, m2)

    def update_fn(updates, state, params, grad_std=None):

        # update adam moments
        m1 = jax.tree.map(lambda g, m: b1*m + (1-b1)*g, updates, state.m1)
        m2 = jax.tree.map(lambda g, m: b2*m + (1-b2)*(g**2), updates, state.m2)

        # scale by adam
        m1_hat = otu.tree_bias_correction(m1, b1, state.step+1)
        m2_hat = otu.tree_bias_correction(m2, b2, state.step+1)
        updates = jax.tree.map(lambda m, v: m / (jnp.sqrt(v) + eps), m1_hat, m2_hat)

        # add weight decay
        updates = jax.tree.map(lambda g, p: g + weight_decay * p, updates, params)

        # scale by lr
        updates = jax.tree.map(lambda g: -learning_rate * g, updates)

        return updates, ScaleByAdamW2State(state.step+1, m1, m2)

    return base.GradientTransformation(init_fn, update_fn)

