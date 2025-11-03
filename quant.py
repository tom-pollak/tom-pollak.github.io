# %%
import torch
from torch import Tensor
# %%

def unpack(x: Tensor, nbits=4):
    """
    Dequantize a 2D array x: [batch, group] -> [batch, group * FPINT]

    FPINT is the number of floating points we can fit in a single integer.
    We can pack any amount of bits less than elemsize, although if it isn't divisible
    by 32 then there will be waste. E.g. for 6-bit we will just use up to 30, not the last 2.
    """
    elemsize_bits = x.element_size() * 8
    FPINT = elemsize_bits // nbits
    B, GROUP = x.shape

    assert not x.is_floating_point(), "x must be int dtype"

    # we have FPINT values to extract, strided by nbits each: [0, 4, 8, 12, 16, 18, ...]
    # therefore we make a tensor of FPINT values to left shift by, and use mask
    # to clip all values out of our 4-bit range.
    shift = torch.arange(FPINT, device=x.device) * nbits
    mask = 2**nbits - 1

    # here we make a new axis of x to expand our dequanted values into.
    # our axis will contain FPINT elems, broadcasted from shift.
    deq_x = (x[..., None] >> shift) & mask # [B, GROUP, FPINT]
    # now let's view our elems flattened.
    return deq_x.reshape(B, GROUP * FPINT).to(torch.uint8)

weight = torch.tensor([[0x01234567, 0x89ABCDEF], [0x9324521A, 0xADEF234A]]).to(torch.int32)
deq_weight = unpack(weight)
deq_weight

# >>> tensor([[ 7,  6,  5,  4,  3,  2,  1,  0, 15, 14, 13, 12, 11, 10,  9,  8],
#               ...])
#
# notice how the values are reversed inside the byte! this is because our shift
# starts at 0, so the first elem is shifted by 0 (aka not at all), so we take the
# least significant byte (LSB), the 7 is the first value to be unpacked.

# %%

def pack(x: Tensor, nbits=4):
    """
    x: [batch, d] -> [batch, d // FPINT (aka GROUP)]
    """
    FPINT = 32 // nbits # int32
    B, D = x.shape
    GROUP = D // FPINT

    assert (x >= 0).all() and (x < (1 << nbits)).all()
    assert not x.is_floating_point(), "x must be int dtype"
    assert D % FPINT == 0, f"{D=}, {FPINT=}"

    # Our D dimension has GROUP * FPINT packed elements, so let's expand this
    x = x.to(torch.int32).reshape(B, GROUP, FPINT)

    # Now we can apply the reverse transform from unpack:
    # note that the mask is _before_ the shift now.
    shift = torch.arange(FPINT, device=x.device, dtype=torch.int32) * nbits
    mask = 2**nbits - 1
    q_x = (x & mask) << shift
    # sum is OR over nibbles, since all slots are disjoint
    return q_x.sum(-1, dtype=torch.int32)  # B GROUP FPINT -> B GROUP


# I made this design so that you should be able to pack any amount of bits, as long as
# group is divisible. So we can do 6 bit just as easily:

nbits = 6
batch, group = 4, 20 # group must be divisible by nbits
x = torch.randint(0, 1 << nbits, (batch, group), dtype=torch.uint8) # random nbit nums
packed = pack(x, nbits=nbits)
unpacked = unpack(packed, nbits=nbits)
print(f"{x=}\n{packed=}\n{unpacked=}")

torch.testing.assert_close(unpacked, x, rtol=0., atol=0.)

# %%
# Now let's go onto a slightly faster algo for int4:
# Here what we're doing is instead of broadcasting a new dimension for each int32 to unpack,
# we view the int32 as 4x uint8.
#
# So in the above example 0x01234567 this gives: [0x67, 0x45, 0x23, 0x01] (little-endian)
# Now we have another problem though, we need to unpack each nibble into a single float.
#
# We can do this by masking `lo` and `hi` of each nibble to give us:
# hi = [6, 4, 2, 0]; lo = [7, 5, 3, 1]
# then interleaving to make [7, 6, 5, 4, 3, 2, 1, 0]

def unpack_4bit(x: Tensor):
    assert x.is_contiguous()
    FPINT = 8 # 32 / 4
    B, G = x.shape
    x = x.view(torch.uint8)
    lo = x & 0xF # bottom nibble
    hi = x >> 4  # top nibble
    q_x = torch.stack((lo, hi), dim=-1) # interleave [lo0,hi0,lo1,hi1,...]
    return q_x.reshape(B, G * FPINT)

nbits = 4
batch, group = 4, 32 # group must be divisible by nbits
x = torch.randint(0, 1 << nbits, (batch, group), dtype=torch.uint8) # random nbit nums
packed = pack(x, nbits=nbits)
unpacked = unpack_4bit(packed)
torch.testing.assert_close(unpacked, x, rtol=0., atol=0.)

# %%
import triton
import triton.language as tl

if torch.cuda.is_available():
    torch._logging.set_logs(output_code=True) # type: ignore
    torch.compile(unpack_4bit)(packed.cuda())
    torch.compile(unpack)(packed.cuda())


## unpack_4bit
@triton.jit
def triton_poi_fused___rshift___bitwise_and_stack_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = xindex // 2
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full([1], 15, tl.uint8)
    tmp7 = tmp5 & tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 2, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr0 + (x1), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full([1], 4, tl.uint8)
    tmp15 = tmp13 >> tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp10, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp9, tmp17)
    tl.store(out_ptr0 + (x2), tmp18, xmask)

## unpack
def triton_poi_fused___rshift____to_copy_arange_bitwise_and_mul_unsqueeze_view_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_ptr0 + (x2 // 8), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.int64)
    tmp2 = 4*((x0 % 8))
    tmp3 = tmp1 >> tmp2
    tmp4 = tl.full([1], 15, tl.int64)
    tmp5 = tmp3 & tmp4
    tmp6 = tmp5.to(tl.uint8)
    tl.store(out_ptr0 + (x2), tmp6, xmask)


def pack_4bit(x: Tensor):
    pass
