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
    warmup_steps = int(c.warmup_frac * c.num_microbtach_steps)
    cooldown_steps = int(c.cooldown_frac * c.num_microbtach_steps)
    stable_steps = c.num_microbtach_steps - warmup_steps - cooldown_steps

    # warmup
    schedules = [
        optax.linear_schedule(init_value=0, end_value=c.peak_learning_rate, transition_steps=warmup_steps)
    ]

    # stable
    schedules.append(
        optax.constant_schedule(value=c.peak_learning_rate)
    )

    # decay
    if c.cooldown_type == 'cosine':
        schedules.append(
            optax.cosine_decay_schedule(init_value=c.peak_learning_rate, decay_steps=cooldown_steps)
        )
    elif c.cooldown_type == 'linear':
        schedules.append(
            optax.linear_schedule(init_value=c.peak_learning_rate, end_value=0, transition_steps=cooldown_steps)
        )
    else:
        raise NotImplementedError(f'Unsupported decay type: {c.cooldown_type}')

    return optax.join_schedules(schedules, boundaries=[warmup_steps, warmup_steps+stable_steps])


def hparam_str_to_schedule(s, tokens_per_microbatch):
    """
    - 's' must be either a scalar or schedule in the format 'b1:v1;b2:v2...', where 'b:v' is boundary:value
    - example inputs: '1', '5.2', '0:5;100:10;200:20'
    - the boundaries are measured in number of tokens seen
    - the returned schedule is piecewise constant
    """
    
    # case 1: scalar value
    if (not isinstance(s, str)) or (';' not in s):
        return float(s)

    # case 2: picewise constant schedule
    # assuming (transition:value) format, e.g. '0:5;1_000:10'

    # get transition boundaries and values
    values = {}
    for item in s.split(';'):
        boundary, value = item.split(':')
        boundary = int(boundary)
        values[boundary] = float(value)
    
    # get initial value
    if 0 not in values:
        raise ValueError('Schedule must include a value for step 0')
    init_value = values.pop(0)

    # create schedule function
    def schedule(microbatch_step):
        v = init_value
        tokens_seen = microbatch_step * tokens_per_microbatch
        for boundary, value in sorted(values.items()):
            # if tokens_seen >= boundary, update value; otherwise keep current value
            update = (tokens_seen >= boundary)
            v = value * update + (1 - update) * v
        return v
        
    return schedule


def get_optimizer(c: OmegaConf, tokens_per_microbatch: int):
    learning_rate_fn = get_learning_rate_schedule(c)
    multistep_wrapper = multistep.SingleSteps if c.grad_accumulation_steps==1 else multistep.MultiSteps
    assert c.optimizer == 'adamw2'
    optimizer = optax.inject_hyperparams(
        lambda learning_rate, t1, r, weight_decay, grad_acc_steps: 
            multistep_wrapper(
                adamw2(
                    learning_rate=learning_rate,
                    tokens_per_opt_step=grad_acc_steps*tokens_per_microbatch,
                    t1=t1,
                    r=r,
                    eps=c.eps,
                    weight_decay=weight_decay,
                ),
                grad_acc_steps,
            )
        )(learning_rate=learning_rate_fn,
          t1=hparam_str_to_schedule(c.adam_t1, tokens_per_microbatch),
          r=hparam_str_to_schedule(c.adam_t2_t1_ratio, tokens_per_microbatch),
          weight_decay=hparam_str_to_schedule(c.weight_decay, tokens_per_microbatch),
          grad_acc_steps=hparam_str_to_schedule(c.grad_accumulation_steps, tokens_per_microbatch),
        )
    return optimizer


class ScaleByAdamW2State(NamedTuple):
    step: jax.Array
    m1: base.Params # ema of g
    m2: base.Params # ema of g**2


def adamw2(
        learning_rate: float,
        tokens_per_opt_step: int,
        t1: float = 2_000_000, # β1 decay half-life in num. tokens
        r: float = 0.999, # ratio of β2/β1 decay half-life
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
    ) -> base.GradientTransformation:

    # convert half-lives to decay values
    t2 = t1 * r # β2 decay half-life
    b1 = utils.halflife_to_decay(t1, tokens_per_opt_step) # β1 decay coefficient
    b2 = utils.halflife_to_decay(t2, tokens_per_opt_step) # β2 decay coefficient
    # ema1_decay = utils.halflife_to_decay(c.ema1_halflife, tokens_per_opt_step)
    # ema2_decay = utils.halflife_to_decay(c.ema2_halflife, tokens_per_opt_step)

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

