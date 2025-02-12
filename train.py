import os
import jax
import jax.numpy as jnp
import optax
import wandb
import operator as op
import orbax.checkpoint as ocp
import data, utils
import model as model_lib
import optimizer as optimizer_lib
from flax import nnx
from tqdm.auto import tqdm
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from omegaconf.dictconfig import DictConfig
from flax.nnx.variablelib import VariableState

def loss_fn(model, batch):
  x, y, weights = data.get_in_out(batch)
  logits = model(x)
  losses = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
  mean_loss = jnp.sum(losses * weights) / weights.sum()
  return mean_loss


def accuracy_fn(model, batch):
  x, y, weights = data.get_in_out(batch)
  logits = model(x)
  hits = logits.argmax(-1) == y
  acc = jnp.sum(hits * weights) / weights.sum()
  return acc


# training step (full batch training)
def train_step_full_batch(opt_graphdef, opt_state, batch):
  optimizer = nnx.merge(opt_graphdef, opt_state)
  loss, grads = nnx.value_and_grad(loss_fn)(optimizer.model, batch)
  optimizer.update(grads)
  _, opt_state = nnx.split(optimizer)
  return opt_state, loss


# training step (single sample training)
def train_step_single_sample(opt_graphdef, opt_state, batch):
  def step_fn(i, args):
    opt_state, loss_sum = args
    optimizer = nnx.merge(opt_graphdef, opt_state)
    loss, grads = nnx.value_and_grad(loss_fn)(optimizer.model, batch[i, None])
    optimizer.update(grads)
    _, opt_state = nnx.split(optimizer)
    return opt_state, loss_sum+loss
  opt_state, loss_sum = jax.lax.fori_loop(0, len(batch), step_fn, (opt_state, 0.))
  return opt_state, loss_sum/len(batch)


def update_moving_averages(step, params, swa, emas, ema_decays):
  """note: step must start at 1, not 0"""

  # update emas
  def update_ema(e, p):
    d = jnp.expand_dims(ema_decays, range(1, e.ndim))
    return d * e + (1 - d) * p[None]
  emas = jax.tree.map(update_ema, emas, params)

  # update SWA
  if swa is not None:
    def update_swa(s, p):
      return (1-1/step)*s + (1/step)*p
    swa = jax.tree.map(update_swa, swa, params)

  return swa, emas


def welford_update(step, x, mean, m2):
  """https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm"""
  delta = x - mean
  mean += delta / step
  delta2 = x - mean
  m2 += delta * delta2
  return mean, m2


def get_ds_loss(model, ds):
  def step_fn(i, loss):
    return loss + loss_fn(model, ds[i])
  loss_sum = jax.lax.fori_loop(0, len(ds), step_fn, 0.)
  return loss_sum / len(ds)


def get_ds_accuracy(model, ds):
  def step_fn(i, acc):
    return acc + accuracy_fn(model, ds[i])
  accuracy_sum = jax.lax.fori_loop(0, len(ds), step_fn, 0.)
  return accuracy_sum / len(ds)


def get_ds_loss_and_grad(model, ds):
  params = nnx.state(model, nnx.Param)
  grad_stats = jax.tree.map(lambda x: (0.*x, 0.*x), params) # (mean, m2)
  def step_fn(i, args):
    loss_sum, grad_stats = args
    loss, grad = nnx.value_and_grad(loss_fn)(model, ds[i])
    grad_stats = jax.tree.map(lambda g, stats: welford_update(i+1, g, *stats), grad, grad_stats)
    return loss_sum+loss, grad_stats
  loss_sum, grad_stats = jax.lax.fori_loop(0, len(ds), step_fn, (0., grad_stats))
  grad_mean = jax.tree.map(lambda _, stats: stats[0], params, grad_stats)
  grad_std = jax.tree.map(lambda _, stats: jnp.sqrt(stats[1]/len(ds)), params, grad_stats)
  loss_mean = loss_sum / len(ds)
  return loss_mean, grad_mean, grad_std


@jax.jit
def compute_metrics_eval(model_graphdef, opt_state, ds_valid, swa, emas, ema_decays, n_param):
  metrics = {}
  params = opt_state.model

  # base loss
  model = nnx.merge(model_graphdef, params)
  eval_loss, grad_mean, grad_std = get_ds_loss_and_grad(model, ds_valid)
  metrics['eval_loss'] = eval_loss
  metrics['eval_acc'] = get_ds_accuracy(model, ds_valid)

  # gradient norm (using tree)
  metrics['grad_norm_L2'] = jnp.sqrt(jax.tree.reduce(lambda s, x: s+(x**2).sum(), grad_mean, 0.)) # L2 norm

  # gradient norms (using raveled gradients)
  grad_mean = jax.tree.map(lambda x: jax.lax.with_sharding_constraint(x.reshape((-1, jax.device_count())), P(None, 'data')), grad_mean)
  grad_mean = jnp.concatenate(jax.tree.leaves(grad_mean)) # [-1, device_count]
  grad_abs = jnp.abs(grad_mean)
  lo, hi = jnp.quantile(grad_abs, jnp.array([0.1, 0.9]), axis=0).mean(1)
  grad_clipped = grad_abs.clip(lo, hi)
  metrics['grad_median'] = jnp.median(grad_abs, axis=0).mean()
  metrics['grad_norm_L2_wins'] = jnp.sqrt(jax.tree.reduce(lambda s, x: s+(x**2).sum(), grad_clipped, 0.))

  # gradient variance
  # to abstract away batch size, we estimate gradient variance of a single sample (rather than a full batch)
  n_batches = len(ds_valid)
  std_batch = jax.tree.reduce(lambda s, x: s+x.sum(), grad_std, 0.) / n_param # average over params
  std_sample = jnp.sqrt(n_batches) * std_batch
  metrics['grad_std'] = std_sample
  metrics['grad_std_normalized_L2'] = std_sample / metrics['grad_norm_L2']
  metrics['grad_std_normalized_wins'] = std_sample / metrics['grad_norm_L2_wins']
  metrics['grad_std_normalized_median'] = std_sample / metrics['grad_median']

  # SWA
  if swa is not None:
    model = nnx.merge(model_graphdef, swa)
    metrics['eval_loss_swa'] = get_ds_loss(model, ds_valid)
    metrics['eval_acc_swa'] = get_ds_accuracy(model, ds_valid)

  # ema
  for i, _ in enumerate(ema_decays):
    ema = jax.tree.map(lambda x: x[i], emas)
    model = nnx.merge(model_graphdef, ema)
    name = f'eval_loss_ema_{i}'
    metrics[name] = get_ds_loss(model, ds_valid)

  # ema projected loss
  # def tree_project(v, s):
  #   """project 'v' onto 's'"""
  #   dot_nd = lambda x, y: jnp.dot(x.flatten(), y.flatten())
  #   tree_dot = lambda x, y: jax.tree.reduce(op.add, jax.tree.map(dot_nd, x, y))
  #   cp = tree_dot(v, s) / tree_dot(s, s) # projection scalar
  #   diff = jax.tree.map(lambda s, v: cp*s - v, s, v) # vector: v -> projection of v onto s
  #   return diff
  # ema1 = jax.tree.map(lambda x: x[0], emas)
  # ema2 = jax.tree.map(lambda x: x[1], emas)
  # v = jax.tree.map(op.sub, params, ema1) # the parameters vector we're projecting
  # s = jax.tree.map(op.sub, ema2, ema1) # we're projecting onto the (e1-e2) line
  # diff = tree_project(v, s)
  # projected_params = jax.tree.map(op.add, params, diff)
  # model = nnx.merge(model_graphdef, projected_params)
  # metrics['eval_loss_ema_diff'] = get_ds_loss(model, ds_valid)

  return metrics


def train_and_evaluate(c: DictConfig):
  """Train loop."""

  # setup
  key = jax.random.key(c.seed)
  key, key_model = jax.random.split(key)

  # datastes
  micro_batch_size, r = divmod(c.train_batch_size, c.opt.grad_accumulation_steps)
  # assert c.opt.grad_accumulation_steps == 1, 'grad. accumulation not implemented'
  get_batch_train, ds_train_size = data.make_ds_loader(c.ds_train_path, c.model.L, c.train_batch_size, c.ds_offset_idx)
  get_batch_valid, ds_valid_size = data.make_ds_loader(c.ds_eval_path, c.model.L, c.eval_batch_size)

  # get number of training steps
  num_train_tokens = c.opt.num_train_tokens or ds_train_size
  tokens_per_train_step = c.train_batch_size * c.model.L
  tokens_per_eval_step = c.eval_batch_size * c.model.L
  c.opt.num_train_steps = num_train_tokens // tokens_per_train_step
  c.eval_num_tokens = c.eval_num_tokens or ds_valid_size
  c.eval_steps = max(1, c.eval_num_tokens // tokens_per_eval_step)
  c.eval_every_steps = max(1, c.eval_every_tokens // tokens_per_train_step)

  # model
  # all devices are aligned across a single mesh axis called 'data'
  # we use FSDP to shard data, model, and optimzier parameters across this axis
  mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), ("data",))
  model = model_lib.create_sharded_model(c.model, mesh, c.ckpt_restore_path)
  model_graphdef = nnx.graphdef(model)
  tx = optimizer_lib.get_optimizer(c.opt) # otax optimizer transform
  optimizer = nnx.Optimizer(model, tx)
  opt_graphdef, opt_state = nnx.split(optimizer)
  batch_sharding = NamedSharding(mesh, P('data')) # data parallelism
  with mesh: ds_valid = jnp.stack([jax.device_put(get_batch_valid(i), batch_sharding) for i in range(c.eval_steps)])

  # initialize metrics
  n_param = utils.get_num_model_params(model)
  params_init = nnx.state(model, nnx.Param)
  swa = params_init if c.opt.track_swa else None
  ema_decays = jnp.array([0.5**(1/(t*c.opt.num_train_steps)) for t in c.opt.ema_halflives])
  emas = jax.vmap(lambda _: params_init)(range(len(ema_decays)))

  # set checkpoint steps
  decay_steps = int(c.opt.decay_frac * c.opt.num_train_steps)
  checkpoint_steps = (
    c.opt.num_train_steps-decay_steps-1, # just before cooldown
    c.opt.num_train_steps-1, # final step
  )

  # define training step
  train_step_fn = train_step_single_sample if c.opt.single_step_training else train_step_full_batch
  @jax.jit
  def train_step(step, opt_state, batch, swa, emas, params_init, n_param):
    # optimizer.model.train()
    opt_state, loss = train_step_fn(opt_graphdef, opt_state, batch)
    swa, emas = update_moving_averages(step+1, opt_state.model, swa, emas, ema_decays)
    lr = opt_state.opt_state.inner_opt_state.hyperparams.learning_rate.value
    param_norm = jax.tree.reduce(lambda s, x: s+jnp.abs(x).sum(), opt_state.model, 0.) / n_param
    param_dist = jax.tree.reduce(op.add, jax.tree.map(lambda x0, x1: jnp.abs(x1-x0).sum(), params_init, opt_state.model)) / n_param
    ema_dist = None if len(ema_decays) < 2 else jax.tree.reduce(op.add, jax.tree.map(lambda x: jnp.abs(x[1]-x[0]).sum(), emas)) / n_param
    metrics = {'train_loss': loss, 'param_distance': param_dist, 'param_norm': param_norm, 'learning_rate': lr, 'ema_dist': ema_dist}
    return opt_state, metrics, swa, emas

  # start wandb
  if c.wandb_project is not None:
    wandb.init(project=c.wandb_project, config=utils.flatten_dict(c), mode=c.wandb_mode, name=c.run_name, dir='/tmp')
    wandb.summary.update(dict(n_param=n_param))

  # training loop
  pending_metrics_train = None
  pending_metrics_valid = None
  pbar = tqdm(range(c.opt.num_train_steps))
  with mesh:
    for step in pbar:
      batch = jax.device_put(get_batch_train(step), batch_sharding)

      # training step
      opt_state, metrics_train, swa, emas = train_step(step, opt_state, batch, swa, emas, params_init, n_param)
      metrics_train |= {'train_tokens_seen': (step+1)*tokens_per_train_step}

      # async logging
      if pending_metrics_train is not None:
        pbar.set_postfix_str(f'loss={pending_metrics_train["train_loss"]:.2f}')
        wandb.log(pending_metrics_train, step-1)
      pending_metrics_train = metrics_train
      if pending_metrics_valid is not None:
        wandb.log(pending_metrics_valid, step-1)
        pending_metrics_valid = None

      # eval step
      if (step % c.eval_every_steps == 0) or ((step+1) == c.opt.num_train_steps):
        pending_metrics_valid = compute_metrics_eval(model_graphdef, opt_state, ds_valid, swa, emas, ema_decays, n_param)
        for i, t in enumerate(c.opt.ema_halflives):
          pending_metrics_valid[f'eval_loss_ema_{t*1000:04.0f}'] = pending_metrics_valid.pop(f'eval_loss_ema_{i}') # rename ema keys for better readability

      # checkpoint
      if c.save_checkpoints and step in checkpoint_steps:
        ckpt_path = os.path.abspath(f'{c.ckpt_save_dir}/{wandb.run.name}/{step}/')
        print(f'saving {ckpt_path}')
        os.makedirs(ckpt_path, exist_ok=True)
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(ckpt_path, nnx.state(model), force=True)

    wandb.log(pending_metrics_train, step)
    wandb.log(pending_metrics_valid, step)
