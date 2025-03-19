# BELLA

![Explaining dots](doc/bella.svg)

BELLA is a deterministic method to explain a given numerical value in a tabular dataset -- for example the prediction of a black-box regressor. The expanations are accurate, general, simple, robust, deterministic, and verifiable. BELLA can provide both factual and counterfactual explanations.

BELLA computes the optimal neighbourhood around the given data point and then trains a linear regression model on this neighborhood. This model can then be used to explain the value of the data point. Since the model has been trained in the input feature space, one can easily change the feature values in this model to see how this affects the outcome.

## How to run BELLA
To run the experiments, open the terminal and run:

```python3 run.py```

## Conditions of use

The code is made available under a [MIT License](https://choosealicense.com/licenses/mit/) by [Nedeljko Radulovic](https://nedrad88.github.io/), [Albert Bifet](https://albertbifet.com/), and [Fabian M. Suchanek](https://suchanek.name). If you use BELLA for scientific purposes, please cite our paper

```
@misc{radulovic2023bellablackboxmodel,
      title={BELLA: Black box model Explanations by Local Linear Approximations}, 
      author={Nedeljko Radulovic and Albert Bifet and Fabian M. Suchanek},
      year={2023},
      eprint={2305.11311},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2305.11311}, 
}
```
