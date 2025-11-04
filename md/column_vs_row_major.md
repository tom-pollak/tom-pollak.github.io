# Column vs Row-major Formats

> $A B \equiv (B^T A^T)^T$

```python
M, N, K = 8, 16, 32
A = torch.randn(M, K, device="cuda", dtype=torch.float32)
B = torch.randn(K, N, device="cuda", dtype=torch.float32)
C = A @ B # m k, k n -> m n

# If we pass these as pointers directly into CUTLASS they will be
# viewed as column-major, (which is simply transposed)
A_c, B_c = A.T.contiguous(), B.T.contiguous() # [K, M], [N, K]
C_c = B_c @ A_c # n k, k m -> n m

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
