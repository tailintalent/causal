## Relation Learning With Minimum Predictive Information Regularization
This repository contains the code for the paper "Discovering Nonlinear Relations with Minimum Predictive Information Regularization" by Wu et al. 2019. It contains the implementation of the proposed Minimum Predictive Information Regularization (MPIR) method and several compared methods, as well as the three experiments of the paper.

## Requirements
- Python 3
- [PyTorch](https://pytorch.org/) >= 0.4.1

Install other required packages by:
```
pip install -r requirements.txt
```

This repository uses the submodule [pytorch_net](https://github.com/tailintalent/pytorch_net) for easy construction and training of neural networks. Initialize this submodule by:
```
git submodule init; git submodule update
```

## Learning
The dataset preparation and relation learning with different methods are via the script [causality/causality_unified_exp.ipynb](https://github.com/tailintalent/causal/blob/master/causality/causality_unified_exp.ipynb) (or its corresponding .py file). All datasets are accompanied inside the datasets/ folder or can be directly generated (synthetic dataset). Several methods are provided inside the causality_unified_exp.ipynb script:

- Our MPIR method
- Mutual information
- Transfer Entropy
- Linear Granger
- Elastic Net
- Causal Influence

The result is saved under the data/ folder as a pickle binary file.
