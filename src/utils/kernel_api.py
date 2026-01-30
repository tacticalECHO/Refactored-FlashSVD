"""
Kernel API Wrapper Layer

Centralizes all kernel invocations to handle API compatibility and provide
a stable interface for all Block implementations.

This layer:
1. Isolates Blocks from direct kernel dependencies
2. Handles kernel API changes defensively
3. Provides consistent error messages
"""

import torch
from typing import Optional


def call_flash_svd_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor],
    Uo: torch.Tensor,
    Vo: torch.Tensor,
    **kwargs
) -> torch.Tensor:
    """
    Wrapper for flash_svd_attention kernel.

    Handles potential API changes in the underlying kernel implementation.

    Args:
        Q: Query tensor [B, H, M, dh]
        K: Key tensor [B, H, M, dh]
        V: Value tensor [B, H, M, dh]
        mask: Attention mask [B, M] or None
        Uo: Output projection U factor
        Vo: Output projection V factor
        **kwargs: Additional kernel-specific arguments

    Returns:
        Attention output tensor
    """
    from flashsvd.kernels import flashsvdattn

    try:
        return flashsvdattn.flash_svd_attention(Q, K, V, mask, Uo, Vo)
    except TypeError as e:
        # Handle potential API changes
        if "mask" in str(e):
            # Kernel may have changed mask handling
            return flashsvdattn.flash_svd_attention(Q, K, V, Uo, Vo)
        raise


def call_flashsvd_ffn_v1(
    x: torch.Tensor,
    U1: torch.Tensor,
    V1: torch.Tensor,
    b1: torch.Tensor,
    U2: torch.Tensor,
    V2: torch.Tensor,
    b2: torch.Tensor,
    **kwargs
) -> torch.Tensor:
    """
    Wrapper for flashsvd_ffn_v1 kernel.

    Args:
        x: Input tensor
        U1: FFN intermediate U factor
        V1: FFN intermediate V factor
        b1: FFN intermediate bias
        U2: FFN output U factor
        V2: FFN output V factor
        b2: FFN output bias
        **kwargs: Additional kernel-specific arguments

    Returns:
        FFN output tensor
    """
    from flashsvd.kernels import flashsvdffnv1

    return flashsvdffnv1.flashsvd_ffn_v1(x, U1, V1, b1, U2, V2, b2)


def call_flashsvd_ffn_v2(
    x: torch.Tensor,
    U1: torch.Tensor,
    V1: torch.Tensor,
    b1: torch.Tensor,
    U2: torch.Tensor,
    V2: torch.Tensor,
    b2: torch.Tensor,
    **kwargs
) -> torch.Tensor:
    """
    Wrapper for flashsvd_ffn (v2) kernel.

    Args:
        x: Input tensor
        U1: FFN intermediate U factor
        V1: FFN intermediate V factor
        b1: FFN intermediate bias
        U2: FFN output U factor
        V2: FFN output V factor
        b2: FFN output bias
        **kwargs: Additional kernel-specific arguments

    Returns:
        FFN output tensor
    """
    from flashsvd.kernels import flashsvdffnv2

    return flashsvdffnv2.flashsvd_ffn(x, U1, V1, b1, U2, V2, b2)


def call_flashsvd_ffn(
    x: torch.Tensor,
    U1: torch.Tensor,
    V1: torch.Tensor,
    b1: torch.Tensor,
    U2: torch.Tensor,
    V2: torch.Tensor,
    b2: torch.Tensor,
    kernel_version: str = "v1",
    **kwargs
) -> torch.Tensor:
    """
    Unified FFN kernel dispatcher.

    Args:
        x: Input tensor
        U1, V1, b1: FFN intermediate factors and bias
        U2, V2, b2: FFN output factors and bias
        kernel_version: "v1" or "v2" (default: "v1")
        **kwargs: Additional kernel-specific arguments

    Returns:
        FFN output tensor
    """
    if kernel_version == "v1":
        return call_flashsvd_ffn_v1(x, U1, V1, b1, U2, V2, b2, **kwargs)
    elif kernel_version == "v2":
        return call_flashsvd_ffn_v2(x, U1, V1, b1, U2, V2, b2, **kwargs)
    else:
        raise ValueError(f"Unknown FFN kernel version: {kernel_version}")
