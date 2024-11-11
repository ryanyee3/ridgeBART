# Author: Ryan Yee
# Purpose: settings for benchmark models baseball study
# Details: 
# Dependencies: 

model = c("cos", "tanh", "relu", "pbart")

# replicates
split = 1:20

settings = expand.grid(
  model = model,
  split = split
)
