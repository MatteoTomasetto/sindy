import pysindy as ps
import numpy as np

np.random.seed(2025)

class SINDy:
    """
    Sparse Identification of Nonlinear Dynamics (SINDy).

    Attributes:
        pair_id (int): Identifier for the data pair to consider.
                
        train_data (np.ndarray): Training data.
        n (int): Number of spatial points.
        m (int): Number of time points.
        x (np.ndarray): Spatial coordinates.
        delta_t (float): Timestep length.

        init_data (np.ndarray): Data initialization for prediction.

        prediction_timesteps (np.ndarray): Prediction timesteps for the model.
    """

    def __init__(self, pair_id, config, train_data, delta_t, init_data = None, prediction_timesteps = None):
        """
        Initialize SINDy with the provided configuration.

        Args:
            pair_id (int): Identifier for the data pair to consider.
            config (Dict): Configuration dictionary containing method and parameters.
            train_data (np.ndarray): Training data.
            delta_t (float): Timestep length
            init_data (np.ndarray): Burn-in data for prediction.
            prediction_timesteps (np.ndarray): Prediction timesteps for the model.
        """     

        self.pair_id = pair_id
        self.equation = "PDE" if config['dataset']['name'] == 'PDE_KS' else "ODE" 

        self.train_data = train_data
        self.n = train_data[0].shape[0]
        self.m = train_data[0].shape[1]
        self.x = 32.0 * np.pi / self.n * np.arange(0, self.n).astype(np.float32) if self.equation == "PDE" else np.arange(0, self.n).astype(np.float32) 
        self.delta_t = delta_t
        self.t = np.arange(self.m) * self.delta_t

        # Define the initial data for model prediction
        if init_data is not None: # Parametric interpolation/extrapolation task
            self.init_data = init_data[:,-1]
        elif self.pair_id == 2 or self.pair_id == 4: # Reconstruction task
            self.init_data = train_data[0][:,0]
        else: # Forecasting task
            self.init_data = train_data[0][:,-1]
        
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


        self.differentiation_method_order = config['model']['differentiation_method_order']
        self.differentiation_method = ps.FiniteDifference(order = self.differentiation_method_order)# drop_endpoints=False)
        #    ("Finite Difference", ps.SINDyDerivative(kind="finite_difference", k=1)),
        #ps.SmoothedFiniteDifference(), ps.SINDyDerivative(kind="savitzky_golay", left=0.5, right=0.5, order=3),
        #ps.SINDyDerivative(kind="spline", s=1e-2)),
        # ("Trend Filtered", ps.SINDyDerivative(kind="trend_filtered", order=0, alpha=1e-2)),
        #("Spectral", ps.SINDyDerivative(kind="spectral")),
        #("Kalman", ps.SINDyDerivative(kind="kalman", alpha=0.05)),

        self.feature_library_order = config['model']['feature_library_order']
        self.feature_library = ps.PolynomialLibrary(degree = self.feature_library_order)#, include_bias=False)
        #self.feature_library = ps.FourierLibrary()
        if self.equation == "PDE":
            self.pde_library = ps.PDELibrary(derivative_order = 4,
                                            spatial_grid = self.x,
                                            include_bias = True,
                                            is_uniform = True,
                                            periodic = True,
                                            include_interaction=True)

            self.feature_library = ps.GeneralizedLibrary([self.feature_library, self.pde_library])

        self.threshold = config['model']['threshold']
        self.optimizer = ps.STLSQ(threshold = self.threshold)#, alpha=1e-5, normalize_columns=True)
        #optimizer = ps.SR3(threshold=7, max_iter=10000, tol=1e-15, nu=1e2,  thresholder='l0', normalize_columns=True)
        #optimizer = ps.SR3(threshold=0.05, max_iter=10000, tol=1e-15, thresholder='l1', normalize_columns=True)
        #optimizer = ps.SSR(normalize_columns=True, kappa=5e-3)
        #optimizer = ps.SSR(criteria='model_residual', normalize_columns=True, kappa=5e-3)
        #optimizer = ps.FROLS(normalize_columns=True, kappa=1e-5)


    def build_data(self) -> np.ndarray:
        """
        Build the training data and concatenate the parameter values, if necessary.

        Returns:
            np.array: Training data.
        """
        
        if len(self.train_data) == 1:
            data = self.train_data[0].T if self.equation == "ODE" else np.expand_dims(self.train_data[0], axis = 2)
        else:
            data = []
            for i in range(len(self.train_data)):
                if self.equation == "ODE":
                    data.append(np.hstack((self.train_data[i].T, self.train_parameters[i] * np.ones((self.m, 1)))))
                else:
                    data.append(np.vstack((self.train_data[i], self.train_parameters[i] * np.ones((1, self.m)))))
            
            self.init_data = np.hstack((self.init_data, self.test_parameters))

        return data


    def fit_model(self) -> ps.pysindy.SINDy:
        """
        Fit the model on the training data.
         
        Returns:
            ps.pysindy.SINDy: SINDy model.
        """
        
        data = self.build_data()
        
        multiple_trajectories = False if self.pair_id < 8 else True

        model = ps.SINDy(differentiation_method = self.differentiation_method,
                        feature_library = self.feature_library,
                        optimizer = self.optimizer)

        model.fit(data, t = self.delta_t, multiple_trajectories = multiple_trajectories)

        return model


    def predict(self) -> np.ndarray:
        """
        SINDy predictions.

        Returns:
            np.ndarray: array of predictions.  
        """
    
        model = self.fit_model()
        
        print(f"Model identified for pair_id {self.pair_id}:")
        model.print()

        predictions = model.simulate(self.init_data, t = self.prediction_timesteps)[:, :self.n]

        return predictions.T
