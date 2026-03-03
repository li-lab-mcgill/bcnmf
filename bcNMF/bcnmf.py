import torch
import numpy as np
from tqdm import tqdm


def nmf_sse(X, K, niter=100):
    """
    NMF with sum of squared error loss as the objective
    :param X: M x N input matrix (numpy array)
    :param K: low rank
    :param adata: annotated X matrix with cluster labels for evaluating ARI
    :param niter: number of iterations to run
    :return:
        1. updated W and H that minimize sum of squared error ||X - WH||^2_F s.t. W,H>=0
        2. niter-by-3 tensor with iteration index, SSE, and ARI as the 3 columns
    """

    # Initialize W and H with random values, ensuring they're on the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M, N = X.shape
    W = torch.rand(M, K, device=device)
    H = torch.rand(K, N, device=device)
    X = torch.from_numpy(np.array(X, dtype="float32"))

    # Ensure X, W, and H are on the correct device
    X = X.to(device)
    W = W.to(device)
    H = H.to(device)

    # Initialize performance tracking array
    perf = torch.zeros((niter, 3), dtype=torch.float32, device=device)

    # COMPLETE THIS FUNCTION

    # SOLUTION STARTS
    for i in range(niter):
        # Update H and W with element-wise multiplication and division
        H = H * (W.T @ X) / (W.T @ W @ H + 1e-10)  # Adding a small constant to avoid division by zero
        W = W * (X @ H.T) / (W @ (H @ H.T) + 1e-10)

        # Calculate the performance metrics
        reconstruction = W @ H
        sse = torch.sum((reconstruction - X) ** 2) / (X.size(0) * X.size(1))
        H_np = H.T.cpu().numpy() if H.is_cuda else H.T.numpy()

        # Store iteration, SSE, and ARI
        perf[i, 0] = i
        perf[i, 1] = sse

        print(f"Iter: {i} .. MSE: {sse:.4f}")
    # SOLUTION ENDS

    return W.cpu().numpy(), H.cpu().numpy(), perf.cpu()



def nmf_poisson(X, K, niter=100):
    """
    NMF with Poisson KL divergence as the objective.
    :param X: M x N input matrix (numpy array)
    :param K: low rank
    :param niter: number of iterations to run
    :return:
        1. Updated W and H that minimize Poisson KL divergence
        2. niter-by-2 tensor with iteration index and Poisson loss as the 2 columns
    """

    # Initialize W and H with random values, ensuring they're on the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M, N = X.shape
    W = torch.rand(M, K, device=device)
    H = torch.rand(K, N, device=device)
    X = torch.from_numpy(np.array(X, dtype="float32"))

    # Ensure X, W, and H are on the correct device
    X = X.to(device)
    W = W.to(device)
    H = H.to(device)

    # Initialize performance tracking array
    perf = torch.zeros((niter, 2), dtype=torch.float32, device=device)

    # Main optimization loop
    for i in range(niter):
        # Element-wise updates for H
        WT1 = W.T @ torch.ones((X.size(0), X.size(1)), device=device)
        X_hat = W @ H
        H = H * (W.T @ (X / (X_hat + 1e-16))) / (WT1 + 1e-16)

        # Element-wise updates for W
        oneHT = torch.ones((X.size(0), X.size(1)), device=device) @ H.T
        X_hat = W @ H
        W = W * ((X / (X_hat + 1e-16)) @ H.T) / (oneHT + 1e-16)
        
        # Ensure X_hat remains positive
        # Calculate Poisson KL divergence
        WH = W @ H
        WH = torch.where(WH > 0, WH, torch.tensor(1e-16, device=device))
        log_likelihood = torch.sum(X * torch.log(WH) - WH) / (X.size(0) * X.size(1))

        # Store iteration and Poisson loss
        perf[i, 0] = i
        perf[i, 1] = log_likelihood

        # Logging for debugging
        print(f"Iter: {i} .. Log likelihood: {log_likelihood:.4f}")

    return W.cpu().numpy(), H.cpu().numpy(), perf.cpu()


def nmf_poisson_minibatch(X, K, niter=100, batch_size=128):
    """
    Improved Mini-batch NMF with Poisson KL divergence using gradient accumulation for W.
    :param X: M x N input matrix (numpy array)
    :param K: Low rank
    :param niter: Number of iterations to run
    :param batch_size: Number of samples per mini-batch
    :return:
        1. Final W and H that minimize Poisson KL divergence
        2. niter-by-2 tensor with iteration index and Poisson loss as the 2 columns
    """

    # Initialize W and H with random values
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M, N = X.shape
    W = torch.rand(M, K, device=device)
    H = torch.rand(K, N, device=device)
    X = torch.from_numpy(np.array(X, dtype="float32")).to(device)

    # Performance tracking
    perf = torch.zeros((niter, 2), dtype=torch.float32, device=device)

    # Mini-batch generator
    def minibatch_generator(data, batch_size):
        indices = torch.randperm(data.size(1))
        for start in range(0, data.size(1), batch_size):
            end = min(start + batch_size, data.size(1))
            yield indices[start:end]

    # Main optimization loop
    for i in range(niter):
        total_log_likelihood = 0
        W_grad_numerator = torch.zeros_like(W, device=device)
        W_grad_denominator = torch.zeros_like(W, device=device)
        print(f"Iter: {i} .. Begin: ")
        # Process mini-batches
        for indices_batch in tqdm(minibatch_generator(X, batch_size)):
            X_batch = X[:, indices_batch]
            H_batch = H[:, indices_batch]

            # Update H for the current mini-batch
            WT1 = W.T @ torch.ones_like(X_batch, device=device)
            X_hat = W @ H_batch
            H_batch = H_batch * (W.T @ (X_batch / (X_hat + 1e-16))) / (WT1 + 1e-16)
            H[:, indices_batch] = H_batch

            # Accumulate gradients for W
            X_hat = W @ H_batch
            W_grad_numerator += (X_batch / (X_hat + 1e-16)) @ H_batch.T
            W_grad_denominator += torch.ones_like(X_batch, device=device) @ H_batch.T

            # Compute log-likelihood for the current mini-batch
            WH = W @ H_batch
            WH = torch.where(WH > 0, WH, torch.tensor(1e-16, device=device))
            batch_log_likelihood = torch.sum(X_batch * torch.log(WH) - WH) / (X_batch.size(0) * X_batch.size(1))
            total_log_likelihood += batch_log_likelihood.item()

        # Update W after processing all mini-batches
        W = W * (W_grad_numerator / (W_grad_denominator + 1e-16))

        # Store iteration performance
        perf[i, 0] = i
        perf[i, 1] = total_log_likelihood
        print(f"Iter: {i} .. Total Log likelihood: {total_log_likelihood:.4f}")

    return W.cpu().numpy(), H.cpu().numpy(), perf.cpu()


def contrastive_nmf_sse(X, Y, K, alpha, niter=100):
    """
    NMF with sum of squared error loss as the objective
    :param X: M x N input matrix (numpy array)
    :param K: low rank
    :param adata: annotated X matrix with cluster labels for evaluating ARI
    :param niter: number of iterations to run
    :return:
        1. updated W and H that minimize sum of squared error ||X - WH||^2_F s.t. W,H>=0
        2. niter-by-3 tensor with iteration index, SSE, and ARI as the 3 columns
    """

    # Initialize W and H with random values, ensuring they're on the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M_X, N_X = X.shape
    M_Y, N_Y = Y.shape
    W = torch.rand(M_X, K, device=device)
    H_X = torch.rand(K, N_X, device=device)
    H_Y = torch.rand(K, N_Y, device=device)
    X = torch.from_numpy(np.array(X, dtype="float32"))
    Y = torch.from_numpy(np.array(Y, dtype="float32"))

    # Ensure X, W, and H are on the correct device
    X = X.to(device)
    Y = Y.to(device)
    W = W.to(device)
    H_X = H_X.to(device)
    H_Y = H_Y.to(device)

    # Initialize performance tracking array
    perf = torch.zeros((niter, 2), dtype=torch.float32, device=device)

    # COMPLETE THIS FUNCTION

    # SOLUTION STARTS
    for i in range(niter):
        # Update H and W with element-wise multiplication and division
        H_X = H_X * (W.T @ X) / (W.T @ W @ H_X + 1e-10)  # Adding a small constant to avoid division by zero
        H_Y = H_Y * (W.T @ Y) / (W.T @ W @ H_Y + 1e-10) 
        W = W * (X @ H_X.T + alpha*W @ (H_Y @ H_Y.T)) / (W @ (H_X @ H_X.T)+ alpha*Y @ H_Y.T + 1e-10)

        # Calculate the performance metrics
        reconstruction_X = W @ H_X
        reconstruction_Y = W @ H_Y
        sse = torch.sum((reconstruction_X - X) ** 2) / (X.size(0) * X.size(1)) + alpha*torch.sum((reconstruction_Y - Y) ** 2) / (Y.size(0) * Y.size(1))

        # Store iteration, SSE, and ARI
        perf[i, 0] = i
        perf[i, 1] = sse

        print(f"Iter: {i} .. MSE: {sse:.4f}")
    # SOLUTION ENDS

    return W.cpu().numpy(), H_X.cpu().numpy(), H_Y.cpu().numpy(), perf.cpu()


def contrastive_nmf_poisson(X, Y, K, alpha, niter=100):
    """
    Contrastive NMF with Poisson KL divergence as the objective.
    :param X: M x N input matrix (numpy array)
    :param Y: Background data matrix (numpy array)
    :param K: Low rank
    :param alpha: Contrastive weight
    :param niter: Number of iterations to run
    :return:
        1. Updated W, H_X, and H_Y that minimize the contrastive objective
        2. niter-by-2 tensor with iteration index and contrastive Poisson loss
    """

    # Initialize W, H_X, and H_Y with random values
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M_X, N_X = X.shape
    M_Y, N_Y = Y.shape
    W = torch.rand(M_X, K, device=device)
    H_X = torch.rand(K, N_X, device=device)
    H_Y = torch.rand(K, N_Y, device=device)
    X = torch.from_numpy(np.array(X, dtype="float32"))
    Y = torch.from_numpy(np.array(Y, dtype="float32"))

    # Ensure X, Y, W, H_X, and H_Y are on the correct device
    X = X.to(device)
    Y = Y.to(device)
    W = W.to(device)
    H_X = H_X.to(device)
    H_Y = H_Y.to(device)

    # Initialize performance tracking array
    perf = torch.zeros((niter, 2), dtype=torch.float32, device=device)

    # Main optimization loop
    for i in range(niter):
        # Update H_X 
        WT1 = W.T @ torch.ones((X.size(0), X.size(1)), device=device)
        WH_X = W @ H_X
        H_X = H_X * (W.T @ (X / (WH_X + 1e-16))) / (WT1 + 1e-16)

        # Update H_Y
        WT1 = W.T @ torch.ones((Y.size(0), Y.size(1)), device=device)
        WH_Y = W @ H_Y
        H_Y = H_Y * (W.T @ (Y / (WH_Y + 1e-16))) / (WT1 + 1e-16)

        # Update W
        WH_X = W @ H_X
        WH_Y = W @ H_Y
        HT1_X = torch.ones((X.size(0), X.size(1)), device=device) @ H_X.T
        HT1_Y = torch.ones((Y.size(0), Y.size(1)), device=device) @ H_Y.T
        W = W * (((X / (WH_X + 1e-16)) @ H_X.T) + alpha * HT1_Y) / ((HT1_X + 1e-16) + alpha * ((Y / (WH_Y + 1e-16)) @ H_Y.T))

        # Calculate contrastive Poisson Log_Likelihood
        WH_X = W @ H_X
        WH_X = torch.where(WH_X > 0, WH_X, torch.tensor(1e-16, device=device))
        WH_Y = W @ H_Y
        WH_Y = torch.where(WH_Y > 0, WH_Y, torch.tensor(1e-16, device=device))
        log_likelihood_X = torch.sum(X * torch.log(WH_X) - WH_X) / (X.size(0) * X.size(1))
        log_likelihood_Y = torch.sum(Y * torch.log(WH_Y) - WH_Y) / (Y.size(0) * Y.size(1))
        contrastive_log_likelihood = (log_likelihood_X - alpha * log_likelihood_Y)

        # Store iteration and contrastive loss
        perf[i, 0] = i
        perf[i, 1] = contrastive_log_likelihood

        # Logging for debugging
        print(f"Iter: {i} .. Contrastive Log Likelihood: {contrastive_log_likelihood:.4f}")

    return W.cpu().numpy(), H_X.cpu().numpy(), H_Y.cpu().numpy(), perf.cpu()





def contrastive_nmf_poisson_minibatch(X, Y, K, alpha, niter=100, batch_size=128):
    """
    Mini-batch Contrastive NMF with Poisson KL divergence.
    :param X: M x N input matrix (numpy array)
    :param Y: Background data matrix (numpy array)
    :param K: Low rank
    :param alpha: Contrastive weight
    :param niter: Number of iterations to run
    :param batch_size: Number of samples per mini-batch
    :return:
        1. Final W, H_X, and H_Y that minimize the contrastive objective
        2. niter-by-2 tensor with iteration index and contrastive Poisson loss
    """

    # Initialize W, H_X, and H_Y with random values
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M_X, N_X = X.shape
    M_Y, N_Y = Y.shape
    W = torch.rand(M_X, K, device=device)
    H_X = torch.rand(K, N_X, device=device)
    H_Y = torch.rand(K, N_Y, device=device)
    X = torch.from_numpy(np.array(X, dtype="float32")).to(device)
    Y = torch.from_numpy(np.array(Y, dtype="float32")).to(device)

    # Initialize performance tracking array
    perf = torch.zeros((niter, 2), dtype=torch.float32, device=device)

    # Define mini-batch generator
    def minibatch_generator(data, batch_size):
        indices = torch.randperm(data.size(1))
        for start in range(0, data.size(1), batch_size):
            end = min(start + batch_size, data.size(1))
            yield indices[start:end]

    # Main optimization loop
    for i in range(niter):
        total_log_likelihood = 0

        # Initialize accumulators for W
        W_grad_numerator = torch.zeros_like(W, device=device)
        W_grad_denominator = torch.zeros_like(W, device=device)
        print(f"Iter: {i} .. Begin:")
        # Mini-batch updates for H_X and W using X
        for indices_batch in minibatch_generator(X, batch_size):
            X_batch = X[:, indices_batch]
            H_X_batch = H_X[:, indices_batch]

            # Update H_X for the current mini-batch
            WT1 = W.T @ torch.ones((X_batch.size(0), X_batch.size(1)), device=device)
            WH_X = W @ H_X_batch
            H_X_batch = H_X_batch * (W.T @ (X_batch / (WH_X + 1e-16))) / (WT1 + 1e-16)
            H_X[:, indices_batch] = H_X_batch  # Save back to global H_X

            # Accumulate gradients for W (from X)
            HT1_X = torch.ones((X_batch.size(0), X_batch.size(1)), device=device) @ H_X_batch.T
            W_grad_numerator += (X_batch / (WH_X + 1e-16)) @ H_X_batch.T
            W_grad_denominator += HT1_X

        # Mini-batch updates for H_Y and W using Y
        for indices_batch in minibatch_generator(Y, batch_size):
            Y_batch = Y[:, indices_batch]
            H_Y_batch = H_Y[:, indices_batch]

            # Update H_Y for the current mini-batch
            WT1 = W.T @ torch.ones((Y_batch.size(0), Y_batch.size(1)), device=device)
            WH_Y = W @ H_Y_batch
            H_Y_batch = H_Y_batch * (W.T @ (Y_batch / (WH_Y + 1e-16))) / (WT1 + 1e-16)
            H_Y[:, indices_batch] = H_Y_batch  # Save back to global H_Y

            # Accumulate gradients for W (from Y)
            HT1_Y = torch.ones((Y_batch.size(0), Y_batch.size(1)), device=device) @ H_Y_batch.T
            W_grad_numerator += alpha * HT1_Y
            W_grad_denominator += alpha * ((Y_batch / (WH_Y + 1e-16)) @ H_Y_batch.T)

        # Update W after processing all mini-batches
        W = W * (W_grad_numerator / (W_grad_denominator + 1e-16))

        # Calculate contrastive Poisson Log Likelihood
        WH_X = W @ H_X
        WH_X = torch.where(WH_X > 0, WH_X, torch.tensor(1e-16, device=device))
        WH_Y = W @ H_Y
        WH_Y = torch.where(WH_Y > 0, WH_Y, torch.tensor(1e-16, device=device))
        log_likelihood_X = torch.sum(X * torch.log(WH_X) - WH_X) / (X.size(0) * X.size(1))
        log_likelihood_Y = torch.sum(Y * torch.log(WH_Y) - WH_Y) / (Y.size(0) * Y.size(1))
        contrastive_log_likelihood = log_likelihood_X - alpha * log_likelihood_Y

        # Store iteration and contrastive loss
        perf[i, 0] = i
        perf[i, 1] = contrastive_log_likelihood.item()

        # Logging for debugging
        print(f"Iter: {i} .. Contrastive Log Likelihood: {contrastive_log_likelihood:.4f}")

    return W.cpu().numpy(), H_X.cpu().numpy(), H_Y.cpu().numpy(), perf.cpu()

def contrastive_nmf_sse_multi(X1, Y1, X2, Y2, K, alpha, niter=100):
    """
    Two-modality contrastive NMF with sum-of-squared-error loss (SSE) as monitoring metric.
    
    We have:
      - Modality 1: X1 (target), Y1 (background)
      - Modality 2: X2 (target), Y2 (background)
    Shared latent codes H_X, H_Y across modalities; modality-specific bases W1, W2.
    
    Args
    ----
    X1: numpy array, shape (M1, N_X)  target data modality 1
    Y1: numpy array, shape (M1, N_Y)  background data modality 1
    X2: numpy array, shape (M2, N_X)  target data modality 2
    Y2: numpy array, shape (M2, N_Y)  background data modality 2
    K:  int, rank
    alpha: float, contrast parameter
    niter: int, number of iterations
    
    Returns
    -------
    W1, W2, H_X, H_Y : numpy arrays with learned factors
    perf            : tensor of shape (niter, 2) with [iter, SSE_metric]
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Shapes
    M1, N_X1 = X1.shape
    M1_y, N_Y = Y1.shape
    M2, N_X2 = X2.shape
    M2_y, N_Y2 = Y2.shape

    assert N_X1 == N_X2, "Target sample counts (N_X) must match across modalities."
    assert N_Y == N_Y2, "Background sample counts (N_Y) must match across modalities."
    N_X = N_X1

    # Convert inputs to torch tensors
    X1 = torch.from_numpy(np.array(X1, dtype="float32")).to(device)
    Y1 = torch.from_numpy(np.array(Y1, dtype="float32")).to(device)
    X2 = torch.from_numpy(np.array(X2, dtype="float32")).to(device)
    Y2 = torch.from_numpy(np.array(Y2, dtype="float32")).to(device)

    # Initialize factors
    W1 = torch.rand(M1, K, device=device)
    W2 = torch.rand(M2, K, device=device)
    H_X = torch.rand(K, N_X, device=device)
    H_Y = torch.rand(K, N_Y, device=device)

    eps = 1e-10

    # Performance tracking: [iter, SSE_metric]
    perf = torch.zeros((niter, 2), dtype=torch.float32, device=device)

    for i in range(niter):
        # ---------- Update H_X ----------
        # H_X <- H_X * (sum_m W_m^T X_m) / (sum_m W_m^T W_m H_X)
        num_HX = W1.T @ X1 + W2.T @ X2
        den_HX = W1.T @ (W1 @ H_X) + W2.T @ (W2 @ H_X) + eps
        H_X = H_X * (num_HX / den_HX)

        # ---------- Update H_Y ----------
        # H_Y <- H_Y * (sum_m W_m^T Y_m) / (sum_m W_m^T W_m H_Y)
        num_HY = W1.T @ Y1 + W2.T @ Y2
        den_HY = W1.T @ (W1 @ H_Y) + W2.T @ (W2 @ H_Y) + eps
        H_Y = H_Y * (num_HY / den_HY)

        # ---------- Update W1 ----------
        # W1 <- W1 * (X1 H_X^T + alpha W1 H_Y H_Y^T) / (W1 H_X H_X^T + alpha Y1 H_Y^T)
        HX_HT = H_X @ H_X.T          # K x K
        HY_HT = H_Y @ H_Y.T          # K x K

        num_W1 = X1 @ H_X.T + alpha * (W1 @ HY_HT)
        den_W1 = (W1 @ HX_HT) + alpha * (Y1 @ H_Y.T) + eps
        W1 = W1 * (num_W1 / den_W1)

        # ---------- Update W2 ----------
        # W2 <- W2 * (X2 H_X^T + alpha W2 H_Y H_Y^T) / (W2 H_X H_X^T + alpha Y2 H_Y^T)
        num_W2 = X2 @ H_X.T + alpha * (W2 @ HY_HT)
        den_W2 = (W2 @ HX_HT) + alpha * (Y2 @ H_Y.T) + eps
        W2 = W2 * (num_W2 / den_W2)

        # ---------- Compute SSE-style monitoring metric ----------
        # Note: this is D(X1)+alpha D(Y1) + D(X2)+alpha D(Y2), not the contrastive loss.
        recon_X1 = W1 @ H_X
        recon_Y1 = W1 @ H_Y
        recon_X2 = W2 @ H_X
        recon_Y2 = W2 @ H_Y

        mse_X1 = torch.sum((recon_X1 - X1) ** 2) / (X1.size(0) * X1.size(1))
        mse_Y1 = torch.sum((recon_Y1 - Y1) ** 2) / (Y1.size(0) * Y1.size(1))
        mse_X2 = torch.sum((recon_X2 - X2) ** 2) / (X2.size(0) * X2.size(1))
        mse_Y2 = torch.sum((recon_Y2 - Y2) ** 2) / (Y2.size(0) * Y2.size(1))

        sse_metric = mse_X1 + alpha * mse_Y1 + mse_X2 + alpha * mse_Y2

        perf[i, 0] = i
        perf[i, 1] = sse_metric

        print(f"Iter: {i} .. SSE metric: {sse_metric:.4f}")

    return (
        W1.detach().cpu().numpy(),
        W2.detach().cpu().numpy(),
        H_X.detach().cpu().numpy(),
        H_Y.detach().cpu().numpy(),
        perf.cpu(),
    )




    