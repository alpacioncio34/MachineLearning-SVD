# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python (ML_python_312)
#     language: python
#     name: ml_python_312
# ---

# %% [markdown]
# # 1 Intuition on SVD

# %%
import numpy as np
from numpy.linalg import svd as svd
from numpy.linalg import matrix_rank as matrix_rank
import matplotlib.pyplot as plt

# %load_ext autoreload
# %autoreload 2
from a03_helper import *

# %%
M1 = np.array(
    [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
)

M2 = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 2, 1, 2, 0],
        [0, 2, 1, 2, 0],
        [0, 2, 1, 2, 0],
        [0, 0, 0, 0, 0],
    ]
)

M3 = np.array([[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]])

M4 = np.array(
    [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
    ]
)

M5 = np.array(
    [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
    ]
)

M6 = np.array(
    [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]
)


# %% [markdown]
# ## 1a

# %%
# Your part

# %% [markdown]
# ## 1b

# %%
# YOUR PART

rank_M1 = matrix_rank(M1)
print(f"M1 Rank is {rank_M1}")

U,s,Vt = np.linalg.svd(M1)
plot_matrix(U)
print(f"Main diagonal values of SV are {s}")
plot_matrix(Vt)

# M2
rank_M2 = matrix_rank(M2)
print(f"M2 Rank is {rank_M2}")

U,s,Vt = np.linalg.svd(M2)
plot_matrix(U)
print(f"Main diagonal values of SV are {s}")
plot_matrix(Vt)

# M3
rank_M3 = matrix_rank(M3)
print(f"M3 Rank is {rank_M3}")

U,s,Vt = np.linalg.svd(M3)
plot_matrix(U)
print(f"Main diagonal values of SV are {s}")
plot_matrix(Vt)


# M4
rank_M4 = matrix_rank(M4)
print(f"M4 Rank is {rank_M4}")

U,s,Vt = np.linalg.svd(M4)
plot_matrix(U)
print(f"Main diagonal values of SV are {s}")
plot_matrix(Vt)


# M5
rank_M5 = matrix_rank(M5)
print(f"M5 Rank is {rank_M5}")

U,s,Vt = np.linalg.svd(M5)
plot_matrix(U)
print(f"Main diagonal values of SV are {s}")
plot_matrix(Vt)


# M6
rank_M6 = matrix_rank(M6)
print(f"M6 Rank is {rank_M6}")

U,s,Vt = np.linalg.svd(M6)
plot_matrix(U)
print(f"Main diagonal values of SV are {s}")
plot_matrix(Vt)


# %% [markdown]
# ## 1c

# %%
# You can use the functions svdcomp and plot_matrix from a03_helper.py to visualize the results.
# YOUR PART
U,s,Vt = svd(M5)

# 2. Extract the largest singular value ($\sigma_1$) and its corresponding vectors
sigma_1 = s[0]
u_1 = U[:, 0:1]     # First left singular vector (column)
v_1_T = Vt[0:1, :]  # First right singular vector transposed (row)

# 3. Construct the rank-1 approximation: A_1 = sigma_1 * u_1 * v_1^T
M5_rank1 = sigma_1 * (u_1 @ v_1_T)

# 4. Visualize the original matrix vs. the rank-1 approximation
plot_matrix(M5)
plot_matrix(M5_rank1)

# %% [markdown]
# ## 1d

# %%
# Another method to compute the rank is matrix_rank.
# YOUR PART
U,s,Vt = svd(M6)
print(len([e for e in s if e > 1e-10]))

rank = matrix_rank(M6)
print(f"M6 has a rank of {rank}")

# %%
