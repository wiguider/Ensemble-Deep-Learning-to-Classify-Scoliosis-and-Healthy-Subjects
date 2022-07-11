import os
import pickle
import random
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, ReLU, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, Adagrad
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.random import set_seed
from sklearn.metrics import balanced_accuracy_score

# Seed value
# Apparently you may use different seed values at each stage
seed_value= 0
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set the `tensorflow` pseudo-random generator at a fixed value
set_seed(seed_value)


class EnsembleNNClassifier():
    """EnsembleNNClassifier represents an Ensemble Model of Neural Networks 
    for binary classification. 
    """
    def __init__(self, 
                 input_shape,
                 layers_units,
                 n_members=5):
        """EnsembleNNClassifier represents an Ensemble Model of Neural Networks 
    for binary classification.

        Args:
            input_shape (Tuple): Shape of the input data. (27,)
            layers_units (Tuple): Tuple representing the number of units in each layer. (64, 64,)
            n_members (int, optional): Number of members in the ensemble model. Defaults to 5.
        """
        self.input_shape = input_shape
        self.layers_units = layers_units
        self.num_hidden_layers = len(layers_units)
        self.n_members = n_members
        self.members = []
        self.score = 0
        self._fitted = False
        if len(self.members)> 0:
            self._fitted = True
        self._build_ensemble()
        
        
    def _build_ensemble(self):
        """Builds the ensemble model.
        """
        for model_index in range(self.n_members):
            model_number = model_index + 1
            self._build_model(model_number)
            
    def compile_ensemble(self):
        """Compiles all models in the ensemble.
        """
        for member in self.members:
            self._compile_model(member)

    def fit_evaluate(self,x_train,y_train, x_test, y_test, epochs=100, batch_size=8, verbose=0):
        """Trains all the models in the ensemble for a fixed number of epochs on the training data,
        and evaluates its performance on the test data 

        Args:
            x_train: Input data for training.
            y_train: Target data for training.
            x_test: Input data for testing.
            y_test: Target data for testing. 
            epochs (int, optional): number of training epochs. Defaults to 100.
            batch_size (int, optional): batch size. Defaults to 8.
            verbose (int, optional): Verbosity mode. Defaults to 0.
        """
        if self._fitted:
            raise Exception('The model is already fitted')
        #fit all models
        self.members = []
        self._build_ensemble()
        self.compile_ensemble()

        self.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

        #evaluate ensemble
        score = self.evaluate(x_test, y_test)
        if verbose>0:
            print('> %.3f' % score)
        self.score = score

    def fit(self, x_train, y_train, epochs=100, batch_size=8):
        """Trains all the models in the ensemble for a fixed number of epochs (iterations on a dataset).

        Args:
            x_train (Input data): It could be:
                - A Numpy array (or array-like), or a list of arrays
                    (in case the model has multiple inputs).
                - A TensorFlow tensor, or a list of tensors
                    (in case the model has multiple inputs).
                - A dict mapping input names to the corresponding array/tensors,
                    if the model has named inputs.
                - A `tf.data` dataset. Should return a tuple
                    of either `(inputs, targets)` or
                    `(inputs, targets, sample_weights)`.
                - A generator or `keras.utils.Sequence` returning `(inputs, targets)`
                    or `(inputs, targets, sample_weights)`.
                - A `tf.keras.utils.experimental.DatasetCreator`, which wraps a
                    callable that takes a single argument of type
                    `tf.distribute.InputContext`, and returns a `tf.data.Dataset`.
                    `DatasetCreator` should be used when users prefer to specify the
                    per-replica batching and sharding logic for the `Dataset`.
                    See `tf.keras.utils.experimental.DatasetCreator` doc for more
                    information.
            y_train (Target data.): Like the input data `x_train`,
                it could be either Numpy array(s) or TensorFlow tensor(s).
                It should be consistent with `x` (you cannot have Numpy inputs and
                tensor targets, or inversely). If `x` is a dataset, generator,
                or `keras.utils.Sequence` instance, `y` should
                not be specified (since targets will be obtained from `x`).
            epochs (int, optional): An epoch is an iteration over the entire `x` and `y`
                data provided. The model is not trained for a number of iterations
                given by `epochs`, but merely until the epoch of index `epochs` 
                is reached. Defaults to 100.
            batch_size (int, optional): Number of samples per gradient update. Defaults to 8.
        """
        if self._fitted:
            raise Exception('The model is already fitted')
        
        self.members = []
        self._build_ensemble()
        self.compile_ensemble()
        for model in self.members:
            self._fit_model(model,  x_train, y_train, 
                            epochs=epochs, 
                            batch_size=batch_size)
        self._fitted = True

    def predict(self, X):
        """Generates output predictions for the input samples.

        Args:
            X: Input samples.

        Returns:
            Numpy array(s) of predictions.
        """
        return np.round(self.predict_members(X))

    #evaluate ensemble model
    def evaluate(self, x_test, y_test):
        """Evaluates the ensemble model's predictions 
        using balanced_accuracy_score.

        Args:
            x_test: Input data for testing.
            y_test: Target data for testing.

        Returns:
            float: balanced_accuracy_score
        """
        #make prediction
        y_pred = self.predict(x_test)
        #calculate accuracy
        return balanced_accuracy_score(y_test, y_pred)
    
    def predict_members(self, X):
        """Generates the class probabilities of the input samples X.

        Args:
            X: Input samples.

        Returns:
            Numpy array(s) of the class probabilities of the input samples.
        """
        if self._fitted:
            y_hats = [model.predict(X) for model in self.members]
            y_hats = np.array(y_hats)
            # mean of predictions
            predictions = np.median(y_hats, axis=0)
            return predictions
        raise Exception("The model should be fitted to make predictions")
    

    def _build_model(self, model_number=1):
        """Builds a single feed forward neural nework.

        Args:
            model_number (int, optional): The number of the model. Defaults to 1.
        """
        # add the input layer
        inputs = self._add_model_input(model_number)
        # add hiddden layers
        hidden_layers = self._add_hidden_layers(inputs)
        # add the output layer
        outputs = self._add_model_outputs(hidden_layers)
        # add the model to the ensemble
        self.members.append(Model(inputs=inputs, outputs=outputs, name=f"member_{model_number}"))

    def _add_model_input(self, model_number):
        """Instantiates a Keras tensor and sets the input shape of the encoder. 

        Args:
            model_number (int): The number of the model.

        Returns:
             A `tensor`.
        """
        return Input(shape=self.input_shape, name=f"model_{model_number}_input")

    def _add_hidden_layers(self, inputs):
        """Creates all the neural blocks in the neural network.

        Args:
            inputs (tensor): The input layer.

        Returns:
            _type_: the graph of layers in the neural netwok.
        """
        x = inputs
        for layer_index in range(self.num_hidden_layers):
            x = self._add_hidden_layer(layer_index, x)
        return x

    def _add_hidden_layer(self, layer_index, x):
        """Adds a neural block to the graph of layers, consisting of a dense
        layer + ReLU + batch normalization. 

        Args:
            layer_index (int): index of the layer to create.
            x (_type_): the graph of layers already in the neural netwok.

        Returns:
            _type_: the graph of layers in the neural netwok 
            including the newly added layer.
        """
        layer_number = layer_index + 1
        hidden_layer = Dense(self.layers_units[layer_index], name=f"dense_layer_{layer_number}")
        x = hidden_layer(x)
        x = ReLU(name=f"relu_{layer_number}")(x)
        x = BatchNormalization(name=f"batch_normalization_{layer_number}")(x)
        return x 
    
    def _add_model_outputs(self, x):
        """Adds an output layer to the graph of layers in the network.

        Args:
            x (tensor): the graph of layers in the network.

        Returns:
            tensor: The graph of layers in the network plus the output layer.
        """
        logits = Dense(units=1, name=f"model_logits")(x)
        output = Activation('sigmoid', name=f"sigmoid_layer")(logits)
        return output
    
    def _compile_model(self, model, learning_rate=0.0001):
        """Configures the model for training.
        Args:
            model (Model): The model to configure.
            learning_rate (float, optional): The learning rate of the optimizer. Defaults to 0.0001.
        """
        optimizer = Adam(learning_rate=learning_rate)
        accuracy = BinaryAccuracy()
        model.compile(loss='binary_crossentropy', 
                      optimizer=optimizer, 
                      metrics=[accuracy])
        
    def _fit_model(self,model, x_train, y_train, epochs=100, batch_size=8):
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        Args:
            model: Model to train.
            x_train (Input data): It could be:
                - A Numpy array (or array-like), or a list of arrays
                    (in case the model has multiple inputs).
                - A TensorFlow tensor, or a list of tensors
                    (in case the model has multiple inputs).
                - A dict mapping input names to the corresponding array/tensors,
                    if the model has named inputs.
                - A `tf.data` dataset. Should return a tuple
                    of either `(inputs, targets)` or
                    `(inputs, targets, sample_weights)`.
                - A generator or `keras.utils.Sequence` returning `(inputs, targets)`
                    or `(inputs, targets, sample_weights)`.
                - A `tf.keras.utils.experimental.DatasetCreator`, which wraps a
                    callable that takes a single argument of type
                    `tf.distribute.InputContext`, and returns a `tf.data.Dataset`.
                    `DatasetCreator` should be used when users prefer to specify the
                    per-replica batching and sharding logic for the `Dataset`.
                    See `tf.keras.utils.experimental.DatasetCreator` doc for more
                    information.
            y_train (Target data.): Like the input data `x_train`,
                it could be either Numpy array(s) or TensorFlow tensor(s).
                It should be consistent with `x` (you cannot have Numpy inputs and
                tensor targets, or inversely). If `x` is a dataset, generator,
                or `keras.utils.Sequence` instance, `y` should
                not be specified (since targets will be obtained from `x`).
            epochs (int, optional): An epoch is an iteration over the entire `x` and `y`
                data provided.
                The model is not trained for a number of iterations
                given by `epochs`, but merely until the epoch
                of index `epochs` is reached. Defaults to 100.
            batch_size (int, optional): Number of samples per gradient update. Defaults to 8.

        Returns:
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs.
        """
        monitor = EarlyStopping(monitor='binary_accuracy', min_delta=1e-3, 
                        patience=5, verbose=1, mode='auto')
        #fit model
        model.fit(x_train, y_train, 
                  epochs=epochs, 
                  batch_size=batch_size,
                  shuffle=True, 
                  callbacks=[monitor],
                  verbose=0)
        return model

    def _create_folder_if_it_doesnt_exist(self, folder_path):
        """Creates a folder if it does not exist in the given path.

        Args:
            folder_path (str): Path of the folder.
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
    def _save_parameters(self, folder_path):
        """Saves the parameters of the model.

        Args:
            folder_path (str): Path of the folder where the parameters will be saved.
        """
        parameters = [
            self.input_shape,
            self.layers_units,
            self.n_members
        ]
        save_path = os.path.join(folder_path, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)
            
    def _save_weights(self, folder_path):
        """Saves the weights of the model.

        Args:
            folder_path (str):  Path of the folder where the weights will be saved.
        """
        for idx, model in zip(range(self.n_members), self.members):
            file_path = os.path.join(folder_path, f"weights_{idx}.h5")
            model.save_weights(file_path)

    def save(self, folder_path="."):
        """Creates a folder if it does not exist in the given path. 
        And, saves the parameters and weights of the model in `folder_path`.

        Args:
            folder_path (str, optional): Path of the folder.
            Defaults to the path of the current directory.
        """
        self._create_folder_if_it_doesnt_exist(folder_path)
        self._save_parameters(folder_path)
        self._save_weights(folder_path)

    def load_weights(self, weights_path):
        """Loads the weights of the model saved in the given path.

        Args:
            weights_path (str): Path of the file where the weights are saved.
        """
        for model in self.members:
            model.load_weights(weights_path)
        
    @classmethod
    def load(cls, folder_path="."):
        """Loads the model from the given folder `folder_path`.

        Args:
            folder_path (str, optional): Path of the folder.
            Defaults to the path of the current directory.

        Returns:
            EnsembleNNClassifier: The model saved in the given folder.
        """
        parameters_path = os.path.join(folder_path, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        ensemble_nn = cls(*parameters)
        
        for idx, model in zip(range(ensemble_nn.n_members), ensemble_nn.members):
            weights_path = os.path.join(folder_path, f"weights_{idx}.h5")
            model.load_weights(weights_path)
        ensemble_nn._fitted = True
        return ensemble_nn
