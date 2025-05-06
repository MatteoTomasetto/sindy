# Sparse Identification of Nonlinear Dynamics

This directory contains an implementation of *SINDy* for [CTF-for-Science](https://github.com/CTF-for-Science).


## Files
- `sindy.py`: Contains the `SINDy` class implementing the model logic based on [pysindy](https://github.com/dynamicslab/pysindy).
- `run.py`: Batch runner script for running the model across multiple sub-datasets in the [CTF-for-Science](https://github.com/CTF-for-Science) framework.
- `config_KS.yaml`: Configuration file for running the model on `PDE_KS` test cases for all sub-datasets.
- `config_Lorenz.yaml`: Configuration file for running the model on `ODE_Lorenz` test cases for all sub-datasets.

The configuration files specify the hyperparameters for running the model with the following structure
```yaml
dataset:
  name: <dataset_name>  # Test case (e.g. PDE_KS, ODE_Lorenz)
  pair_id: 'all'        # Which sub-datasets to consider
model:
  name: SINDy
```

## Usage

In the [CTF-for-Science](https://github.com/CTF-for-Science) framework, the DeepONet model can be tested with the command

```bash
python models/sindy/run.py models/sindy/config_*.yaml
```

## Dependencies


## References
