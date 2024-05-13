library(nlshrink)
library(RcppCNPy)
library(reticulate)

np <- import("numpy")
pastRet <- np$load("C:/Users/lyhen/Downloads/MTP2/pastRet_100_25.npy") # first we save the file in .npy to better manipule in R
for (h in 1:360) { x <- data_r[h,,] - apply(data_r[h,,], 2, mean) NLS_cov <- nlshrink_cov(x, k=1)

# Define the path for saving using paste0 for dynamic file names
file_path <- paste0("C:/Users/lyhen/Downloads/MTP2/cov_NLS/cov_h", h, ".npy")

# Save the cov matrix as a .npy file
np$save(file_path, NLS_cov)
}