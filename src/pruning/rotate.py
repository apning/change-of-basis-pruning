import torch


def compute_rotation(H, norm="l2", num_iters=30) -> torch.Tensor:
    """
    This function finds an orthonormal rotation R that orders feature directions by importance according to an L2 or L1 heuristic

    Args:
        H : torch.Tensor, shape (N, d), this is the embedding matrix

        norm : str
            "l2" is for l2 norm -> leads to PCA based rotation
            "l1" is for l1 norm -> leads to L1-PCA sign-power iteration

        num_iters : int
            Number of iterations for L1-PCA

    Returns:
        R : torch.Tensor, shape (d, d) [out, in]
            The orthonormal rotation matrix.
    """
    H = H.to(torch.float32)
    N, d = H.shape

    # This is code for l2 pca based rotation
    if norm.lower() == "l2":
        # covariance matrix (d, d)
        C = H.T @ H
        # eigvectors in ascending order
        eigvals, eigvecs = torch.linalg.eigh(C)
        # reverse to descending (most important first)
        R = torch.flip(eigvecs, dims=[1])

    # This is for l1 iterative approach
    elif norm.lower() == "l1":
        R = torch.eye(d, device=H.device)
        for _ in range(num_iters):
            # project (N, d)
            Y = H @ R
            # (+1/-1) (N, d)
            S = torch.sign(Y)
            # (d, d)
            G = H.T @ S
            # here we re-orthonormalize
            R, _ = torch.linalg.qr(G)

        # Order columns by L1 norm (highest first)
        Y = H @ R
        # per-basis L1 mass
        l1_norms = torch.sum(torch.abs(Y), dim=0)
        sorted_indices = torch.argsort(l1_norms, descending=True)
        # most-important-first
        R = R[:, sorted_indices]

    else:
        raise ValueError("norm must be 'l1' or 'l2'")

    # We transpose so the matrix's shape is (out, in), matching pytorch conventions
    return R.T
