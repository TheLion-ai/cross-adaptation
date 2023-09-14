Cross-adaptation
==============================
# Description

Cross-adaptation performs domain adaptation across a set of domains of all the available training datasets $\{D_1,D_2,...,D_n\}$ (where $n$ is the number of training datasets) by iteratively changing which dataset is the source and which datasets are the target. It follows the case of multi-target domain adaptation.  The algorithm produces transformed datasets $D_x = \{D_{x_1}, D_{x_2},...,{D_{x_n}}\}$. The resulting datasets $D_x$ that is a concatenation of all transformed datasets $\{D_{x_1}, D_{x_2},...,{D_{x_n}}\}$ can then be paired with its respective labels to train the model $f$. This model can then be successfully used on a new domain not present in the training domains $D$. As shown in the Algorithm \ref{alg:domain-generalization}, this method works irrespective of the domain adaptation method $g$. We can denote the source and target domains as $D_s$ and $D_t$. 

## Algorithm

<img width="555" alt="image" src="https://github.com/TheLion-ai/cross-adaptation/assets/12778421/7f1e7fc0-e9cc-4263-af82-97df8b225ec1">

# Usage
## Installation
Prerequisites
* python = "^3.8,<3.11"
* [Poetry](https://python-poetry.org/docs/)

Install requirements
```
poetry install
```
## Using cross-adaptation
```python
import pandas as pd
from adapt.instance_based import KMM
from sklearn.tree import DecisionTreeClassifier

from cross_adaptation import Adapter

data: {
    "dataset1" : pd.DataFrame(),
    "dataset2" : pd.DataFrame()
} # the data to be adapted, each dataframe should have a column called target
adapt_model = KMM(...) #the method for domain adaptation
classifier = DecisionTreeClassifier(...) #estimator used for the task

adapter = Adapter(
    data,
    adapt_model,
    classifier
)

adapted_dataset = adapter.adapt()

```
## Running experiments

* adjust config in `experiments/config`
* run `experiments/main.py`
  ```bash
    poetry run experiments/main.py
  ```


