# grvmodel
An exploratory development of a Neural Network based Machine Learning algorithm for predicting the outcomes of greyhound races in Victoria.

This algorthim is an adapted impletmeation of the Pytorch Neural Network package for Python.

## Current Models in Rotaion:

### "GRU - WANDB - priced - Fasttrack v5.3 CEL weighted"

Model on full ANZ dataset, using individual GRU cells tied to individual dogs across time.

Features:

  -Cross Entropy Loss Function utalizing the box preference % to weight the classes and deal with imbalance
  -Sum CEL losses of 10 batches and then back propergates loss
  -Runs @~0.3 accuracy with specific tracks running up to 10% profit across 80/20 split
  
Questions to answer/Things to do:
 
  -Add in a confidence/market_price estimator and find lowest delta to return profit
  -Try to vary the minibatch size, and the count of minibatches before backwards pass, could increase accuracy
  -Do a full forward pass of full batch just after backwards pass to updates internal GRU cell states (valid/useless?)
  -Pull out a last validation set (1 Month) and test model profitiabilty outsite train/test split
  
### "GRU - WANDB - priced - Fasttrack v5.4 KL NZ weighted"

Model on small NZ Dataset, benefit of being quick to cycle epochs and all dogs contained within 5 major tracks.

(Also operates on 5% comission from betfair)

Features:

  -KL Loss Function utalizing the box preference % to weight the classes (per box not by winner) and deal with imbalance
  -Sum KL losses of 10 batches and then back propergates loss
  -Runs @~0.25 accuracy with specific tracks running up to 8% profit across 80/20 split
  
Questions to answer/Things to do:
 
  -Add in a confidence/market_price estimator and find lowest delta to return profit
  -Try to vary the minibatch size, and the count of minibatches before backwards pass, could increase accuracy
  -Do a full forward pass of full batch just after backwards pass to updates internal GRU cell states (valid/useless?)
  -Pull out a last validation set (1 Month) and test model profitiabilty outsite train/test split

