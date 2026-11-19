"""Compact weights (bf16/fp16 where autocast rounds anyway, fp32 elsewhere) on a small
network on the CPU: which tensors are rounded, the file reads back exactly, and a model
loaded from it computes under autocast bit for bit what the original computed. Plus the
Hugging Face cache helper. No GPU and no real model needed."""
import os

import pytest

from vibekimus.encoders.compact_weights import CompactWeights, bytes_needed, rounded_names
from vibekimus.encoders.hf_cache import remove_cached_file

torch = pytest.importorskip("torch")
nn = torch.nn

SEED = 0
WIDTH, HEADS, VOCAB, TOKENS = 64, 4, 1000, 12
SIZE_ESTIMATE_SLACK = 16_384   # bytes of the file's header the estimate leaves out
ROUNDED = {"patch.weight", "patch.bias", "qkv.weight", "qkv.bias", "mlp.0.weight", "mlp.0.bias", "mlp.2.weight",
           "mlp.2.bias", "attention.in_proj_weight", "attention.in_proj_bias", "attention.out_proj.weight",
           "attention.out_proj.bias"}
SIXTEEN_BITS = [torch.bfloat16, torch.float16]


class Tiny(nn.Module):
    """The kinds of layers SigLIP 2 has: a patch convolution, a word table, position
    embeddings, layer norms, attention (both open_clip's and a linear-qkv one) and MLPs."""

    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, WIDTH, kernel_size=4, stride=4)
        self.pos = nn.Parameter(torch.randn(1, 16, WIDTH) * 0.02)
        self.words = nn.Embedding(VOCAB, WIDTH)
        self.norm = nn.LayerNorm(WIDTH)
        self.attention = nn.MultiheadAttention(WIDTH, HEADS, batch_first=True)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.mlp = nn.Sequential(nn.Linear(WIDTH, 4 * WIDTH), nn.GELU(), nn.Linear(4 * WIDTH, WIDTH))
        self.register_buffer("positions", torch.arange(TOKENS))   # not a float: stays as it is

    def image(self, pixels):
        x = self.patch(pixels).flatten(2).transpose(1, 2) + self.pos
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        x = x + nn.functional.scaled_dot_product_attention(q, k, v)
        return x + self.mlp(self.norm(x))

    def text(self, tokens):
        x = self.words(tokens)
        x = x + self.attention(self.norm(x), self.norm(x), self.norm(x), need_weights=False)[0]
        return x + self.mlp(self.norm(x))


def outputs(net, pixels, tokens, dtype):
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=dtype):
        return net.image(pixels).float(), net.text(tokens).float()


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(SEED)
    return Tiny().eval()


@pytest.fixture
def compact(net, tmp_path, request):
    """The compact file of `net`, rounded to the 16-bit type the test is parametrized with."""
    dtype = request.param
    weights = CompactWeights(str(tmp_path / f"tiny_{dtype}.safetensors"))
    weights.write(net, dtype, source="test")
    return weights, dtype


def test_rounded_are_convolutions_linear_layers_and_attention_projections(net):
    assert rounded_names(net) == ROUNDED, sorted(rounded_names(net) ^ ROUNDED)


@pytest.mark.parametrize("compact", SIXTEEN_BITS, indirect=True)
def test_rounded_tensors_are_stored_in_16_bits_other_floats_in_float32(compact, net):
    from safetensors import safe_open
    weights, dtype = compact
    with safe_open(weights.path, framework="pt") as file:
        kinds = {name: file.get_tensor(name).dtype for name in file.keys()}
    assert all(kinds[name] == dtype for name in rounded_names(net))
    assert kinds["words.weight"] == torch.float32 and kinds["norm.weight"] == torch.float32
    assert kinds["positions"] == torch.int64, "not a float: unchanged"
    assert weights.rounded_to() == str(dtype).replace("torch.", "")


@pytest.mark.parametrize("compact", SIXTEEN_BITS, indirect=True)
def test_the_size_estimate_matches_the_file(compact, net):
    weights, _dtype = compact
    assert abs(os.path.getsize(weights.path) - bytes_needed(net)) < SIZE_ESTIMATE_SLACK


@pytest.mark.parametrize("compact", SIXTEEN_BITS, indirect=True)
def test_loaded_into_a_float32_model_the_outputs_under_autocast_are_bit_identical(compact, net):
    from safetensors.torch import load_file
    weights, dtype = compact
    loaded = Tiny().eval()
    loaded.load_state_dict(load_file(weights.path))
    pixels, tokens = torch.rand(2, 3, 16, 16), torch.randint(0, VOCAB, (2, TOKENS))
    assert all(torch.equal(a, b) for a, b in zip(outputs(net, pixels, tokens, dtype),
                                                  outputs(loaded, pixels, tokens, dtype)))


@pytest.mark.parametrize("compact", SIXTEEN_BITS, indirect=True)
def test_no_temporary_file_is_left_behind(compact, tmp_path):
    assert not any(name.endswith(".tmp") for name in os.listdir(tmp_path))


def test_removing_a_cached_file_frees_its_size(tmp_path):
    plain = tmp_path / "plain.bin"
    plain.write_bytes(b"y" * 500)
    assert remove_cached_file(str(plain)) == 500 and not plain.exists()


def test_a_cached_file_kept_as_a_link_is_removed_together_with_its_blob(tmp_path):
    blob, link = tmp_path / "blob", tmp_path / "link.bin"
    blob.write_bytes(b"x" * 1000)
    try:
        os.symlink(blob, link)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not allowed here")
    assert remove_cached_file(str(link)) == 1000 and not link.exists() and not blob.exists()
