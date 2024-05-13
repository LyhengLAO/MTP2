library(POET)
library(RcppCNPy)

library(reticulate)

np <- import("numpy")
pastRet <- np$load("C:/Users/lyhen/Downloads/MTP2/pastRet_100_25.npy") # first we save the file in .npy to better manipule in R

for (h in (1:360)){ x <- data_r[h,,] - apply(data_r[h,,],2,mean) Y <- t(x)

res <- POET(Y,5)
cov <- res$SigmaY

file_path <- paste0("C:/Users/lyhen/Downloads/MTP2/cov_POET5/cov_h", h, ".npy")
np$save(file_path, cov)
}