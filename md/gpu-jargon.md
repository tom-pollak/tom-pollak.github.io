# GPU Jargon

## CUDA Jargon

**SM** (Streaming multiprocessor): The GPU equivalent of a CPU core.
Can hold multiple warps (see below) but only one is executed at a time.

- This allows e.g. a warp to issue a load from memory, and while it's
  waiting a different warp is doing compute work. Switching warps can be
  done in a single cycle.

**CTA** (Cooperative thread array, aka **block**): Threads on the same
SM. They can communicate with each other via shared memory & share a L1.

**Warp**: Collection of 32 threads. 4 warps run on an SM at a time. All threads in a warp *must be executed in lockstep* else we get "[warp divergence](https://modal.com/gpu-glossary/perf/warp-divergence)" (bad).

- We want at least 4 warps per block so that no warp on an SM is left idle.

**Thread block cluster**: Group of up to 16 "close" SMs that can share DSHMEM (distributed shared memory). Also known as *GPC* (graphics processing cluster).

## Kernel Programming

### (Conventional) GPU Programming Model

We launch a "grid" of multiple blocks (CTAs) that get allocated
(somewhat arbitrarily) to SMs. We can change the number of blocks to fit
our problem

- E.g: When running a matmul, each block will compute one output tile.
  as the matmul dimensions increase, we launch more blocks.

This is done by the **scheduler**: launching blocks comes with a **small
but non-trivial overhead**, and there might be some repeated work ran by
every block

- E.g: in a convolution kernel, we might load the filter into shared
  memory. Every time a new block is launched, we have to repeat this
  one-time overhead.

### Persistent Kernels

Instead, most modern kernels ignore the scheduler altogether and **view
a GPU as a collection of SMs** (aka core) and launch a **single CTA (aka
block) per SM**.

We call these "persistent kernels" since the CTA will not exit until the
kernel finishes.

- N.B. You can think of Triton programs as CTAs / blocks.

```python
# Standard
@triton.jit
def vector_add(x_ptr):
    pid = tl.program_id(0)
    x = tl.load(x_ptr + pid)
    out = x + 10
    out.store(out_ptr + pid, out)

vector_add[(len(x),)](x, out)

# Persistent
@triton.jit
def vector_add_persistent(x_ptr, out_ptr, N):
    pid, num_sm = tl.program_id(0), tl.num_programs(0)
    for i in range(pid, N, num_sm):
        x = tl.load(x_ptr + i)
        tl.store(out_ptr + off, x + 10)

# launch NUM_SM CTAs, each CTA gets allocated to a single SM.
vector_add_persistent[(NUM_SM,)](x, out, len(x))
```

### Warp Specialization

More resources:

- Motivation for warp specialization: <https://rohany.github.io/blog/warp-specialization/>
- More: <https://ianbarber.blog/2025/02/16/warp-specialization/> | <https://pytorch.org/blog/warp-specialization/>

Precursor: GPUs are not very good at branching instructions. However as
our kernels get more complex we would like the accelerator to do
different things at the same time.

- The conventional method would be kernels launched in different GPU
  streams that can communicate via global memory (kind equivalent to
  IPC).

  - This is clearly slow! We would like different threads in the same
    CTA (with access to the same shared memory, L1 cache).

- Each SM can execute 8 (on Hopper) warps at a time, but can hold up to
  64 "queued" and is *very good at switching warps* (single cycle).

  - This is used for latency hiding, while one warp is waiting for data,
    another can be running compute.

We saw earlier that divergent code paths *inside* a warp is bad (warp
divergence), but the same is not true *between warps*.

- We can calculate our current warp, and branch based on that, avoiding
  warp divergence!

- Warp specialization is often combined with persistent kernels.

```c
__global__ void persistent_producer_consumer_kernel(...) {
    extern __shared__ float smem[];
    int warpId = threadIdx.x / 32; // 32 threads per warp

    for (int tile = blockIdx.x; tile < NTILES; tile += gridDim.x) {
        if (warpId < P) { // P producer warps
            produce_tile(tile, smem); // cp.async/TMA to shared
        } else { // remaining consumer warps
            consume_tile(tile, smem); // MMA on data in shared
        }
        __syncthreads(); // double-buffer, ping-pong, etc.
    }
}
```

- This wasn't as useful until *Hopper:* previously each thread would get
  the same number of registers (255), so there was incentive to
  distribute the load to avoid [register
  pressure](https://modal.com/gpu-glossary/perf/register-pressure).

  - *Hopper* brought **dynamic register allocation**, This isn't a
    problem anymore!

  - In general there seems to be push to get rid of the reliance on
    registers: TMA loading directly into shared mem, Tensor Core memory,
    dynamic register allocation.

- Triton isn't designed for this, each "program" controls a single CTA.
  You can't further subdivide into warps. This is a big reason why you
  can't write FA4 in Triton? Also memory layouts I think.

This can all look a bit convoluted and annoying since now we basically
roll our own scheduler and event loop.

## Memory Semantics

### Memory Hierarchy

<https://www.aleksagordic.com/blog/matmul>

- Register file (RMEM): 256 per thread

- Shared: Per SM, 1KB per block

  - Faster than L1.

- L1: per SM

  - Cache line: 128B, aka **(4B per thread in warp!)**

- Distributed shared (DSMEM), Pooled SMEM of thread block cluster.

- L2: cross-SM

  - Cache line: 128B

  - Good for atomics.

  - Residency control: set L2 cache as part of GMEM

- Global

  - Constant cache, (but also gets cached to L2 easily?)

### Eviction Policy

See <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-eviction-priority-hints> for more info.

Controls in what order to flush cache lines when they are full:

- "" (evict_normal): LRU
- evict_first
- evict_last

Note: `no_allocate` is not supported?

### Cache Modifier

Controls at what stage in the memory hierarchy to cache to. This relates directly to PTX ld / cp / st.

See <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators> for more info.

`volatile=True`: Disable compiler optimizations that could reorder or eliminate the load. Notably different from `.cv`.

#### Load

- "" (default): Allow hardware to pick, same as .ca?
- .ca (all levels): Cache at all levels (L1, L2, Global) with LRU policy
- .cg (global): Bypass L1, cache only at a "global level". Improves cross-SM visibility / L1 thrashing.
- .cv (volatile) Bypass all GPU caches and fetch directly from global.
- .cs (streaming): Store with evict_first and limit cache pollution. Used when "streaming" output data back to memory that you don't intend on re-reading.
- .lu (last use): Indicate this is the last time the variable will be loaded. (add more here)

#### Store

- "" (default): Allow hardware to pick, same as .wb?
- .wb (write-back): TODO: can populate L2 cache lines on write out, maybe L1 too??
- .wt (write-through): TODO
- .cg: Same as load.
- .cs: Same as load

### PTX Examples

#### Load & Store

- `%rN`: 32-bit register
- `%rdN`: 64-bit register (mainly for pointers)
- `[...]`: Memory operand, e.g. a register that holds a byte address: `[%rd0]`
- `a`, `x`: arbitrary variables, I generally use a, b, c for registers and x, y, z for pointers

```ptx
// Load syntax:
// ld.space.type dest, [addr];
ld.global.u32 %r0, [%rd0];  // load 32 bits from global to register

// Store syntax:
// st.space.type [addr], src;
st.shared.f32 [x], a;  // store float32 in register a to shared mem at pointer x

// Cache modifiers: space.{modifier}.type
ld.shared.cg.b32 a, [x];  // cache global
st.global.wt.u8 [x], a;   // cache streaming

// Eviction policy: space.{modifier}.{eviction_policy}.type
ld.global.L1::evict_last.u32 a, [x];  // evict last from L1
ld.global.L1::evict_first.L2::evict_last.u32 a, [x];  // eviction policies are defined seperately for each cache
ld.global.cg.L2::evict_last.u32 a, [x];  // cache global and L2 evict last

// Volatile: {volatile}.space.type
// Note since we're not doing any caching this is not compat with cache modifier / eviction policy
ld.volatile.global.u32 a, [x];  // volatile=True

// Prefetch: space.{modifier}.{eviction_policy}.{prefetch}.type
ld.global.L2::64B.b32 a, [x];  // prefetch 64B into L2 along with the 4B you load

// Vectorized loads: space.{modifier}.{eviction_policy}.{prefetch}.{vectorized}.type
ld.global.v2.u32 {a,b}, [x];  // load into multiple registers

// Combine everything -- load 4 64 bit registers, prefetch 256B into L2. evict_last on L2
ld.global.cg.L2::evict_last.L2::256B.v4.u64 { a,b,c,d }, [x];
```

#### Looking at DeepSeeks undocumented instruction

We're almost ready to look at the infamous DeepSeek "undocumented instruction":

```ptx
ld.global.nc.L1::no_allocate.L2::256B
```

#### Async Copy

<https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async>

Issuing non-blocking copies is important for good perf. This is what the "consumer" does in warp specialization.

```ptx
// Async copy, each copies 16 bytes from global to shared.
cp.async.ca.shared.global [shrd], [gbl], 16;
cp.async.cg.shared.global [shrd+16], [gbl+16], 16;  // .cg modifier
cp.async.wait_all;  // wait for all copies to finish, kind of

// Bundle all previous async into an "async group", which bundles all loads together
cp.async.commit_group;

// 0: wait for all groups to finish. 1: allow newest group to still be in flight.
cp.async.wait_group 0;
```

When multiple threads write to the same memory location, we can get race conditions. The way around this is to use atomic operations, which you can think about as thread-safe, but slower than standard ops.

#### TMA Loads

Asynchronous data transfer between GMEM and SMEM / DSHMEM

TMA loads have high latency (due to address generation overhead? -- look
into)

<https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-memory-access>

### Atomic semantics

There are different types of atomics, denoted in triton with the `sem` param. By default it uses `sem="acq_rel"`:

- Acquire / release: This is a really bad default for most atomics operations! It acts as a memory barrier in the program, no loads / stores can be reordered to before or after it.
- Instead we should use `sem="relaxed"`. This enforces atomic operations on that variable / mem, but doesn't care about the rest of the program.

Available semantic options:

- `acq_rel` (default)
- `relaxed`
- `acquire`
- `release`

## Tensor Cores

Head canon:

```python
import torch
from einops import einsum, rearrange

TS = 32
M, N, K = 256, 256, 1024
A, B = torch.randn(M, K), torch.randn(K, N)

# TMA load: swizzle matrices into 32x32 tiles.
A_tiled = rearrange(A, "(i tm) (k tk) -> i k tm tk", tm=TS, tk=TS)
B_tiled = rearrange(B, "(k tk) (j tn) -> k j tk tn", tn=TS, tk=TS)

# Each thread group computes one tile of output:
i, j = 1, 1
C_ij = torch.zeros(32, 32)  # accumulator

for k in range(K // TS):
    C_ij += A_tiled[i, k] @ B_tiled[k, j]  # Tensor core op

# Batched tile matmul method
C_tiled = einsum(A_tiled, B_tiled, "i k tm tk, k j tk tn -> i j tm tn")
C = rearrange(C_tiled, "i j tm tn -> (i tm) (j tn)")

torch.testing.assert_close(C, A @ B, rtol=0., atol=0.)
torch.testing.assert_close(C_ij, C_tiled[i, j], rtol=1e-4, atol=1e-4)
```

## Triton Features

### tl.range

- <https://ianbarber.blog/2025/05/09/how-does-triton-do-warp-spec/>
- <https://triton-lang.org/main/python-api/generated/triton.language.range.html>
- Enables warp specialization in Triton

### triton.Config

<https://triton-lang.org/main/python-api/generated/triton.Config.html>

- What happened to `num_consumer_groups=2`, `num_buffers_warp_spec=3`?
  - Only trace I can find: <https://github.com/triton-lang/triton/blob/25bf669f82c47a534c8dbdc59063b1ec2b8d231c/docs/meetups/03-12-2025/notes.md?plain=1#L26>

<https://www.aleksagordic.com/blog/matmul>

## Modern Hardware Features

- <https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/#hopper>

### Tensor Memory

This is *not* directly related to TMA

- <https://modal.com/gpu-glossary/device-hardware/tensor-memory>

### 2CTA

### Warpgroup

- To load from tensor memory, we need to use warpgroups. These are
  groups of four warps.

### Thread Block Cluster

- Up to 16 neighboring thread blocks can read and write to others' shared memory via SM-to-SM network.
- More details: <https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md>

#### Column vs row-major formats

```python
M, N, K = 8, 16, 32

A = torch.randn(M, K, device="cuda", dtype=torch.float32)
B = torch.randn(K, N, device="cuda", dtype=torch.float32)
C = A @ B  # m k, k n -> m n

# If we pass these as pointers directly into CUTLASS they will be
# viewed as column-major, (which is simply transposed)
A_c, B_c = A.T.contiguous(), B.T.contiguous()  # [K, M], [N, K]
C_c = B_c @ A_c  # n k, k m -> n m

# This creates `C_c`, shape `[N, M]`, which is equivalent to
# transposed C.

# Now the real trick is that `C_c` is written back to memory in
# column-major format, so what happens if we load it back up in
# row-major PyTorch?

# **It loads `C_c.T`, aka C!**
torch.testing.assert_close(C, C_c.T)

# So all we need to do when using cutlass gemm's with our
# row-major tensors is swap `A` and `B` args
```
