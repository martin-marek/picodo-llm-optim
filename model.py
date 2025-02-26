import os
import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
import orbax.checkpoint as ocp
from omegaconf.dictconfig import DictConfig


class TransformerDecoder(nnx.Module):
    def __init__(self, cfg: DictConfig, rngs: nnx.Rngs):
        self.embed = nnx.Embed(num_embeddings=cfg.V, features=cfg.D, embedding_init=fsdp_init('embedding', cfg.fsdp_enabled), rngs=rngs)
        self.pos_embed = nnx.Embed(num_embeddings=cfg.L, features=cfg.D, embedding_init=fsdp_init('embedding', cfg.fsdp_enabled), rngs=rngs)
        block_fn = nnx.remat(TransformerBlock) if cfg.remat else TransformerBlock
        self.blocks = [block_fn(cfg, rngs) for _ in range(cfg.N)]
        self.out_ln = nnx.LayerNorm(cfg.D, use_bias=False, dtype=cfg.dtype, rngs=rngs)
        
    def __call__(self, x):    # [B, S]
        # Token + positional embedding
        h = self.embed(x) + self.pos_embed(jnp.arange(x.shape[1])[None, ...])    # [B, S, D]
        
        # Transformer blocks
        for block in self.blocks:
            h = block(h)
            
        # Project back to vocabulary
        h = self.out_ln(h)
        return self.embed.attend(h.astype(jnp.float32))    # [B, S, V]


class TransformerBlock(nnx.Module):
    def __init__(self, cfg: DictConfig, rngs: nnx.Rngs):
        self.ln1 = nnx.LayerNorm(cfg.D, use_bias=False, dtype=cfg.dtype, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=cfg.H, in_features=cfg.D, qkv_features=cfg.D, out_features=cfg.D,
            kernel_init=fsdp_init('attn_in_proj', cfg.fsdp_enabled), out_kernel_init=fsdp_init('attn_out_proj', cfg.fsdp_enabled),
            use_bias=False, dtype=cfg.dtype, rngs=rngs, decode=False,
        )
        self.ln2 = nnx.LayerNorm(cfg.D, use_bias=False, dtype=cfg.dtype, rngs=rngs)
        self.mlp = Mlp(cfg, rngs)
        
    def __call__(self, x):    # [B, S, D]
        # Pre-layernorm attention block
        h = self.ln1(x)
        mask = nnx.make_causal_mask(jnp.ones((x.shape[0], x.shape[1])), dtype=x.dtype)
        x = x + self.attn(h, mask=mask)
        
        # Pre-layernorm MLP block
        return x + self.mlp(self.ln2(x))


class Mlp(nnx.Module):
    """Multilayer perceptron."""
    def __init__(self, cfg: DictConfig, rngs: nnx.Rngs):
        kernel_init = fsdp_init('mlp_kernel', cfg.fsdp_enabled)
        self.fc1 = nnx.Linear(in_features=cfg.D, out_features=cfg.F, use_bias=False, kernel_init=kernel_init, dtype=cfg.dtype, rngs=rngs)
        self.fc2 = nnx.Linear(in_features=cfg.F, out_features=cfg.D, use_bias=False, kernel_init=kernel_init, dtype=cfg.dtype, rngs=rngs)
        
    def __call__(self, x):    # [B, S, D]
        h = jax.nn.gelu(self.fc1(x))    # [B, S, F]
        return self.fc2(h)    # [B, S, D]


def fsdp_init(layer_type: str, fsdp_enabled: bool):
    """Initialize weights with optional FSDP partitioning."""
    partition_fn = nnx.with_partitioning if fsdp_enabled else lambda x, _: x
    kernel_init = jax.nn.initializers.xavier_uniform()
    embed_init = jax.nn.initializers.variance_scaling(1.0, 'fan_in', 'normal', out_axis=0)
    match layer_type:
        case "embedding":    # [V, D]
            return partition_fn(embed_init, (None, "data"))
        case "attn_in_proj":    # [D, H, D/H]
            return partition_fn(kernel_init, ("data", None, None))
        case "attn_out_proj":    # [H, D/H, D]
            return partition_fn(kernel_init, (None, None, "data"))
        case "mlp_kernel":    # [D, F]
            return partition_fn(kernel_init, ("data", None))
        case _:
            raise ValueError(f"unrecognized layer type: {layer_type}")


def create_sharded_model(c: DictConfig, mesh: Mesh):
    """https://flax.readthedocs.io/en/latest/guides/flax_gspmd.html"""
    # TODO: add rng key

    # initialize sharded model without putting it on a single device
    @nnx.jit
    def initialize_sharded_model():
        model = TransformerDecoder(c.model, rngs=nnx.Rngs(c.seed)) # unsharded at this moment
        state = nnx.state(model) # the model's state, a pure pytree
        pspecs = nnx.get_partition_spec(state) # get annotations from state
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(model, sharded_state) # the model is sharded now
        return model
    with mesh:
        model = initialize_sharded_model()

    # optioanlly load existing checkpoint
    if c.ckpt_restore_path is not None:
        checkpointer = ocp.StandardCheckpointer()
        state = checkpointer.restore(os.path.abspath(c.ckpt_restore_path), target=nnx.state(model))
        nnx.update(model, state)

    return model
