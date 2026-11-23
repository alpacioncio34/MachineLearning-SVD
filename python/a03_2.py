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
# # 2 The SVD on Weather Data

# %%
import numpy as np
from numpy.linalg import svd as svd
from numpy.linalg import matrix_rank as matrix_rank
import pandas as pd
import matplotlib.pyplot as plt

# %load_ext autoreload
# %autoreload 2
from a03_helper import *

# %%
# The date is being loaded via `a03_helper.py`

# Plot the coordinates
plot_xy(lon, lat)


# %% [markdown]
# ## 2a

# %%
# YOUR PART
# Normalize the data to z-scores. Store the result in X.
# For this, complete the code in `a03_helper.py`.

# %%
# Plot histograms of attributes
nextplot()
X.hist(ax=plt.gca())
plt.show()

# %% [markdown]
# ## 2b

# %%
# Compute the SVD of the normalized climate data and store it in variables U,s,Vt. What
# is the rank of the data?
# %%
U, s, Vt = svd(X)
# Numerical rank based on a tolerance threshold
threshold = 1e-10 * s[0]
Z_rank = np.sum(s > threshold)
print(f"Our data has rank {Z_rank}")
# %% [markdown]
# ## 2c

# %%
# Here is an example plot.
plot_xy(lon, lat, U[:, 0])

# %%
# For interpretation, it may also help to look at the other component matrices and
# perhaps use other plot functions (e.g., plot_matrix).
# YOUR PART
plot_xy(lon, lat, U[:, 0])
plot_xy(lon, lat, U[:, 1])
plot_xy(lon, lat, U[:, 2])
plot_xy(lon, lat, U[:, 3])
plot_xy(lon, lat, U[:, 4])

# %% [markdown]
# ## 2d

# %%
# Here is an example.
plot_xy(U[:, 0], U[:, 1], lat - np.mean(lat))
plot_xy(U[:, 0], U[:, 2], lat - np.mean(lat))
plot_xy(U[:, 1], U[:, 2], lat - np.mean(lat))
plot_xy(U[:, 2], U[:, 3], lat - np.mean(lat))

# %% [markdown]
# ## 2e

# %%
# 2e(i) Guttman-Kaiser
# YOUR PART
def k_th_variance(sigma,N):
    return sigma**2/(N-1)

# Select largest Z such that the variance >1 for all k <=Z
i = len(s) - 1
while i >= 0 and k_th_variance(s[i], N=X.shape[0]) <= 1:
    i -= 1

Z = i + 1
print(Z)


# %%
# 2e(ii) 90% squared Frobenius norm
# YOUR PART
frob_norm_sq = np.sum(s**2)
cum_energy = np.cumsum(s**2) / frob_norm_sq
Z = np.argmax(cum_energy >= 0.9) + 1
print(Z)

# %%
# 2e(iii) Scree test
# YOUR PART
import matplotlib
A = X
U, S, V = np.linalg.svd(A) 
num_vars = len(S)
eigvals = S**2 / np.sum(S**2)  # NOTE (@amoeba): These are not PCA eigenvalues. 
                               # This question is about SVD.

fig = plt.figure(figsize=(8,5))
sing_vals = np.arange(num_vars) + 1
plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
#I don't like the default legend so I typically make mine like below, e.g.
#with smaller fonts and a bit transparent so I do not cover up data, and make
#it moveable by the viewer in case upper-right is a bad place for it 
leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3, 
                 shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                 markerscale=0.4)
leg.get_frame().set_alpha(0.4)
plt.show()

# %%
# 2e(iv) entropy
# YOUR PART
p = s**2 / np.sum(s**2)
p = p[p > 0]  # Prevent log(0)
E = -np.sum(p * np.log(p))

print(f"Normalized/Shannon entropy: {E}")
# Effective dimension / Z based on entropy (perplexity)
Z = int(np.ceil(np.exp(E)))
print(Z)

# %%
# 2e(v) random flips
# Random sign matrix: np.random.choice([-1,1], X.shape)
# YOUR PART
rand_s = np.array([np.linalg.svd(X * np.random.choice([-1, 1], X.shape), compute_uv=False) for _ in range(100)])
Z = np.sum(s > np.percentile(rand_s, 95, axis=0))
print(Z)

# %% [markdown]
# ## 2f

# %%
# Here is the empty plot that you need to fill (one line per choice of Z: RSME between
# original X and the reconstruction from size-Z SVD of noisy versions)
# YOUR PART
nextplot()
plt.plot()
plt.xlabel(r"Noise level ($\epsilon$)")
plt.ylabel("Reconstruction RMSE vs. original data")
