import pysindy as ps
import numpy as np
from typing import List, Tuple, Dict, Optional

np.random.seed(2025)

class SINDy:
    """
    Sparse Identification of Nonlinear Dynamics (SINDy).

    Attributes:
        pair_id (int): Identifier for the data pair to consider.
                
        train_data List[np.ndarray]: Training data.
        n (int): Number of spatial points.
        m (int): Number of time points.
        delta_t (float): Timestep length.
        t (np.ndarray): Time array.

        init_data (np.ndarray): Data initialization for prediction.

        reduction (bool): Whether to use data reduction.
        POD_modes (int): Number of POD modes to use for data reduction.
        
        prediction_timesteps (np.ndarray): Prediction timesteps for the model.

        train_parameters (np.ndarray): Training parameters for the model.
        test_parameters (np.ndarray): Testing parameters for the model.

        differentiation_method (ps.SINDyDerivative): Differentiation method for SINDy.
        feature_library: Feature library for SINDy.
        threshold (float): Threshold for the optimizer.
        optimizer: Optimizer for SINDy.
    """

    def __init__(self, pair_id: int, config: Dict, train_data: List[np.ndarray], delta_t: float, init_data: Optional[np.ndarray] = None, prediction_timesteps: Optional[np.ndarray] = None):
        """
        Initialize SINDy with the provided configuration.

        Args:
            pair_id (int): Identifier for the data pair to consider.
            config (Dict): Configuration dictionary containing method and parameters.
            train_data List[np.ndarray]: Training data.
            delta_t (float): Timestep length
            init_data (np.ndarray): Burn-in data for prediction.
            prediction_timesteps (np.ndarray): Prediction timesteps for the model.

        Raises:
            ValueError: If differentiation method, feature library or optimizer is invalid.
        """     

        self.pair_id = pair_id
        
        self.train_data = train_data
        self.n = train_data[0].shape[0]
        self.m = train_data[0].shape[1]
        self.delta_t = delta_t
        self.t = np.arange(self.m) * self.delta_t

        # Define the initial data for model prediction
        if init_data is not None: # Parametric interpolation/extrapolation task
            self.init_data = init_data[:,-1]
        elif self.pair_id == 2 or self.pair_id == 4: # Reconstruction task
            self.init_data = train_data[0][:,0]
        else: # Forecasting task
            self.init_data = train_data[0][:,-1]

        self.reduction = True if config['dataset']['name'] == 'PDE_KS' else False 
        self.POD_modes = config['model']['POD_modes']

        self.prediction_timesteps = prediction_timesteps

        # Define the model parameters
        if pair_id == 8:
            self.train_parameters =  np.array([1.,2.,4.]) if len(self.train_data) > 2 else np.array([1.,3.])
            self.test_parameters = np.array([3.]) if len(self.train_data) > 2 else np.array([2.])
        elif pair_id == 9:
            self.train_parameters =  np.array([1.,2.,3.]) if len(self.train_data) > 2 else np.array([1.,2.])
            self.test_parameters = np.array([4.]) if len(self.train_data) > 2 else np.array([3.])
        else:
            self.train_parameters = None
            self.test_parameters = None

        if config['model']['differentiation_method'] == 'finite_difference':
            self.differentiation_method = ps.SINDyDerivative(kind = "finite_difference", k = config['model']['differentiation_method_order'])
        elif config['model']['differentiation_method'] == 'savitzky_golay':
            self.differentiation_method = ps.SINDyDerivative(kind = "savitzky_golay", order = config['model']['differentiation_method_order'], left = 0.5, right = 0.5)
        elif config['model']['differentiation_method'] == 'spline':
            self.differentiation_method = ps.SINDyDerivative(kind = "spline", s = 1e-2)
        elif config['model']['differentiation_method'] == 'trend_filtered':
            self.differentiation_method = ps.SINDyDerivative(kind = "trend_filtered", order = config['model']['differentiation_method_order'], alpha = 1e-2)
        elif config['model']['differentiation_method'] == 'spectral':
            self.differentiation_method = ps.SINDyDerivative(kind = "spectral")
        elif config['model']['differentiation_method'] == 'kalman':
            self.differentiation_method = ps.SINDyDerivative(kind = "kalman")
        else:
            raise ValueError("Select a valid differentiation method. The following are available: 'finite_difference', 'savitzky_golay', 'spline', 'trend_filtered', 'spectral', 'kalman'.")

        if config['model']['feature_library'] == 'polynomial':
            self.feature_library = ps.PolynomialLibrary(degree = config['model']['feature_library_order'])
        elif config['model']['feature_library'] == 'Fourier':
            self.feature_library = ps.FourierLibrary(n_frequencies = config['model']['feature_library_order'])
        elif config['model']['feature_library'] == 'mixed':
            self.feature_library = ps.GeneralizedLibrary(
                [ps.PolynomialLibrary(degree = config['model']['feature_library_order']),
                ps.FourierLibrary(n_frequencies = config['model']['feature_library_order'])]
            )
        else:
            raise ValueError("Select a valid feature library. The following are available: 'polynomial', 'Fourier', 'mixed'.")

        self.threshold = config['model']['threshold']
        self.alpha = config['model']['alpha']
        if config['model']['optimizer'] == 'STLSQ':
            self.optimizer = ps.STLSQ(threshold = self.threshold, alpha = self.alpha, normalize_columns = True)
        elif config['model']['optimizer'] == 'SR3':
            self.optimizer = ps.SR3(threshold = self.threshold, normalize_columns = True)
        elif config['model']['optimizer'] == 'SSR':
            self.optimizer = ps.SSR(alpha = self.alpha, normalize_columns = True)
        elif config['model']['optimizer'] == 'FROLS':
            self.optimizer = ps.FROLS(alpha = self.alpha, normalize_columns = True)
        else:
            raise ValueError("Select a valid optimizer. The following are available: 'STLSQ', 'SR3', 'SSR', 'FROLS'.")

    
    def compress_data(self) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """
        Compress the training data with Singular Values Decomposition.

        Returns:
            List[np.ndarray]: Reduced training data.
            np.ndarray: Reduced burn-in data.
            np.ndarray: Left singular values from Singular Values Decomposition.
        """

        full_data = np.concatenate(self.train_data, axis = 1)
                
        U, S, _ = np.linalg.svd(full_data, full_matrices = False)

        residual_energy = np.sum(S[:self.POD_modes]**2) / np.sum(S**2)
        if residual_energy <= 0.9:
            print(f"Warning: The residual energy of the SVD is {residual_energy} <= 0.9 for rank {self.POD_modes}. This may indicate that the rank is too low.")
        elif np.isclose(residual_energy, 1.0):
            print(f"Warning: The residual energy of the SVD is {residual_energy} for rank {self.POD_modes}. This may indicate that the rank is too high.")

        reduced_data = []
        for i in range(len(self.train_data)):
            reduced_data.append(U[:, :self.POD_modes].T @ self.train_data[i])     

        reduced_init_data = U[:, :self.POD_modes].T @ self.init_data if self.init_data is not None else None

        return reduced_data, reduced_init_data, U
    

    def build_data(self, train_data: List[np.ndarray], init_data: np.ndarray) -> np.ndarray:
        """
        Build the training data and concatenate the parameter values, if necessary.

        Args:
            train_data (List[np.ndarray]): Training data.
            init_data (np.ndarray): Burn-in data for prediction.

        Returns:
            np.ndarray: Training data with parameters, if necessary.
            np.ndarray: Burn-in data with parameters, if necessary.
        """
        
        if len(train_data) == 1:
            output_train_data = train_data[0].T
            
            output_init_data = init_data

        else:
            output_train_data = []
            for i in range(len(train_data)):
                output_train_data.append(np.hstack((train_data[i].T, self.train_parameters[i] * np.ones((self.m, 1)))))
            
            output_init_data = np.hstack((init_data, self.test_parameters))

        return output_train_data, output_init_data


    def fit_model(self, train_data: List[np.ndarray]) -> ps.pysindy.SINDy:
        """
        Fit the model on the training data.

        Args:
            train_data List[np.ndarray]: Training data.

        Returns:
            ps.pysindy.SINDy: SINDy model.
        """
                
        multiple_trajectories = False if self.pair_id < 8 else True

        model = ps.SINDy(differentiation_method = self.differentiation_method,
                        feature_library = self.feature_library,
                        optimizer = self.optimizer)

        model.fit(train_data, t = self.delta_t, multiple_trajectories = multiple_trajectories, library_ensemble = True, ensemble = True, quiet=True)

        return model


    def predict(self) -> np.ndarray:
        """
        SINDy predictions.

        Returns:
            np.ndarray: array of predictions.  
        """
    
        if self.reduction:
            train_data, init_data, U = self.compress_data()
        else:
            train_data = self.train_data
            init_data = self.init_data
            U = None

        train_data, init_data = self.build_data(train_data, init_data)

        model = self.fit_model(train_data)
        
        print(f"Model identified for pair_id {self.pair_id}:")
        model.print()

        try:
            predictions = model.simulate(init_data, t = self.prediction_timesteps)
        except: # If integrator fails, consider the static dynamical system \dot{x} = 0 as default          
            print(f"Integrator failed pair_id {self.pair_id}. Using static dynamical system.")
            predictions = np.tile(init_data, (len(self.prediction_timesteps), 1))

        if self.reduction is False:
            predictions = predictions[:, :self.n].T
        else:
            predictions = U[:, :self.POD_modes] @ predictions[:, :self.POD_modes].T

        return predictions
