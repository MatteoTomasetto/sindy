# Sparse Identification of Nonlinear Dynamics

This directory contains an implementation of *Sparse Identification of Nonlinear Dynamics* for [CTF-for-Science](https://github.com/CTF-for-Science).

Sparse Identification of Nonlinear Dynamics (SINDy)* [1] is an algorithm designed to identify nonlinear dynamical systems $\dfrac{d}{dt}𝙭(t) = 𝙛(𝙭(t))$ from time-series data. Sparsity promoting strategies are considered in order to obtain interpretable dynamical systems with few active terms in the governing equations, capable of accurately extrapolating beyond training data. Specifically, given the matrices $X$ and $\frac{d}{dt} X$ collecting, respectively, the time-series $X_{i,j} = x_i(t_j)$ and $\frac{d}{dt} X_{i,j} = \frac{d}{dt}x_i(t_j)$ for $i=1,...,n$ and $j = 1,...,m$, the dynamical system $\dfrac{d}{dt}𝙭(t) = 𝙛(𝙭(t))$ is approximated through

$$
\dfrac{d}{dt} X = \Theta(X) \Xi
$$

where $\Theta(X)$ is a library of regressions terms (polynomials or trigonometric functions are typically considered) and $\Xi$ are the corresponding coefficients, which are determiend through linear regression. To promote sparsity, the Least Absolute Shrinkage and Selection Operator (LASSO) or Sequentially Thresholded Least SQuares (STLSQ) are typically considered to determine the coefficient values.

<br />
<p align="center" width="65%">
  <img width=100% src="./sindy.png" >
  <br />
</p>
<br />

In particular we leverage Ensemble-SINDy [2], which robustify the SINDy algorithm through boostrap aggregating (bagging) strategies.

## Files
- `sindy.py`: Contains the `SINDy` class implementing the model logic based on [pysindy](https://github.com/dynamicslab/pysindy).
- `run.py`: Batch runner script for running the model across multiple sub-datasets in the [CTF-for-Science](https://github.com/CTF-for-Science) framework.
- `run_opt.py`: Batch runner script for running the model across multiple sub-datasets with hyperparameter tuning in the [CTF-for-Science](https://github.com/CTF-for-Science) framework.
- `optimize_parameters.py`: Script for tuning the model hyperparameters
- `config/config_Lorenz.yaml`: Configuration file for running the model on the `Lorenz` test cases for all sub-datasets.
- `config/config_KS.yaml`: Configuration file for running the model on the `Kuramoto-Sivashinsky` test cases for all sub-datasets.
- `config/optimal_config_Lorenz_*.yaml`: Configuration file for running the model on the `Lorenz` test cases with optimal hyperparameters for every sub-datasets.
- `config/optimal_config_KS_*.yaml`: Configuration file for running the model on the `Kuramoto-Sivashinsky` test cases with optimal hyperparameters for every sub-datasets.
- `tuning_config/config_Lorenz_*.yaml`: Configuration file for tuning the model hyperparameters on the `Lorenz` test cases for every sub-datasets.
- `tuning_config/config_KS_*.yaml`: Configuration file for tuning the model hyperparameters on the `Kuramoto-Sivashinsky` test cases for every sub-datasets.
 
The configuration files in the `config` folder specify the hyperparameters for running the model with the following structure
```yaml
dataset:
  name: <dataset_name>  # Test case (e.g. PDE_KS, ODE_Lorenz)
  pair_id: <pair_id>    # Which sub-datasets to consider (e.g. [1, 2, 3], 'all')
model:
  name: SINDy
  POD_modes: <number_POD_modes>                                    # Number of POD modes for dimensionality reduction
  differentiation_method: <differentiation_method>                 # Differentiation method to employ
  differentiation_method_order: <differentiation_method_order>     # Order of the differentiation method
  feature_library: <feature_library>                               # Library functions to consider
  feature_library_order: <feature_library_order>                   # Order of the library functions
  optimizer: <optimizer>                                           # Optimizer to employ to fit the coefficients
  threshold: <threshold>                                           # Threshold value to sparsify the coefficients in the optimizer
  alpha: <alpha>                                                   # Regularization parameter in the optimizer
```

The configuration files in the `tuning_config` folder specify, instead, the possible hyperparameter values to explore while tuning them. 

## Usage

In the [CTF-for-Science](https://github.com/CTF-for-Science) framework, the SINDy model can be tested with the command

```bash
python models/sindy/run.py models/sindy/config_*.yaml
```

## Dependencies
- numpy
- pysindy

[pysindy]([https://github.com/lululxvi/deepxde](https://github.com/dynamicslab/pysindy)) can be installed through the following command
```bash
pip install pysindy
```

## References
[1] S.L. Brunton, J.L. Proctor, J.N. Kutz, *Discovering Governing Equations from Data by Sparse Identification of Nonlinear Dynamical Systems*. Proceedings of the National Academy of Sciences 113 (15): 3932–37 (2016). [https://doi.org/10.1073/pnas.1517384113](https://doi.org/10.1073/pnas.1517384113)

[2] U. Fasel, J.N. Kutz, B.W. Brunton, S.L. Brunton, *Ensemble-SINDy: Robust sparse model discovery in the low-data, high-noise limit, with active learning and control*, Proc. R. Soc. A. 47820210904 (2022). [http://doi.org/10.1098/rspa.2021.0904](http://doi.org/10.1098/rspa.2021.0904)

