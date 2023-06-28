import os

import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

###########################################################################################################
## HELPER FUNCTIONS
###########################################################################################################


def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.
    Returns:
      `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))

    return x * cdf


class AdapterPooler(tf.keras.layers.Layer):
    @tf.keras.dtensor.utils.allow_initializer_layout
    def __init__(self, adapter_dim, init_scale=1e-3, shared_weights=True):
        super().__init__()
        self.adapter_dim = adapter_dim
        self.initializer = tf.keras.initializers.TruncatedNormal(stddev=init_scale)
        if shared_weights:
            self.pooler_layer = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    self.adapter_dim, kernel_initializer=self.initializer
                )
            )
        else:
            self.pooler_layer = tf.keras.layers.LocallyConnected1D(
                self.adapter_dim, 1, 1, kernel_initializer=self.initializer
            )

    def call(self, inputs) -> tf.Tensor:
        """Input shape expected to be (batch_size, 386, 1024, 24)
        Call reshapes tensor into (batch_size * 386, 24, 1024)
        Apply pooler_layer to input with gelu activation
        """

        sequence_dim = inputs.shape[1]
        embedding_dim = inputs.shape[2]
        encoder_dim = inputs.shape[3]

        # Combine batch and sequence length dimension
        X = tf.reshape(inputs, [-1, embedding_dim, encoder_dim])

        # Move encoder_dim to axis = 1
        X = tf.transpose(X, (0, 2, 1))

        X = self.pooler_layer(X)
        X = gelu(X)

        # Regenerate shape
        X = tf.transpose(X, (0, 2, 1))
        X = tf.reshape(X, [-1, sequence_dim, self.adapter_dim, encoder_dim])

        return X


class MeanConcat(tf.keras.layers.Layer):
    def __init__(self, units=1):
        super().__init__()

        # Will only work currently with units = 1
        self.units = 1

    def build(self, input_shape):
        self.last_axis = len(input_shape) - 1

    def call(self, inputs):
        return tf.reduce_mean(inputs, self.last_axis)


###########################################################################################################
## CLASS CONTAINING SPAN ANNOTATION MODELS
###########################################################################################################
class QnAModels(object):
    def __init__(self, **kwargs):
        self.__GPU_count = len(tf.config.list_physical_devices("GPU"))

    ######################################################
    ### Private Methods
    ######################################################

    # validate required input parameter values aren't set to None
    @staticmethod
    def __require_params(**kwargs):
        needed_args = [key for key, value in kwargs.items() if value is None]
        if len(needed_args) > 0:
            raise ValueError(
                "If running in training, must specify following outputs: %s"
                % (", ".join(needed_args))
            )

        return

    def __verbose_print(self, model, model_name, input_shape, opt, loss, metrics):
        print("".join(["\n", "*" * 100, "\nModel Details\n", "*" * 100, "\n"]))
        print(f"Model Name: {model_name}")
        print(
            f"Optimizer Details:  name = {opt.get_config()['name']},  learning rate = {opt.get_config()['learning_rate']}"
        )
        print(
            f"Loss Details:  name = {loss.get_config()['name']}, from_logits = {loss.get_config()['from_logits']}"
        )
        print(f"Input Shape: {tuple(input_shape)}")
        print(f"Metrics: {''.join(metrics)}")
        print("*" * 100)
        print(model.summary())
        print("*" * 100, "\n")

        return

    ######################################################
    ### Public Methods
    ######################################################

    # /////////////////////////////////////////////////////
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    ### Sample Model
    # /////////////////////////////////////////////////////
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    def get_sample_model(
        self, input_shape=(386, 1024, 26), gpu_device="/gpu:0", verbose=True
    ):
        r"""Returns the sample model for QnA Span Annotation task.
            adapter pooler layer inspired by Houlsby et al. 2019 : https://arxiv.org/abs/1902.00751v2

        Args:
            input_shape (tuple, optional): Shape of the input tensor. Defaults to (1, 1024, 26).
            gpu_device (str, optional): If GPU devices are available, defines which one to utilize. Defaults to "/gpu:0".
            verbose (bool, optional): Log details to console. Defaults to True.

        Returns:
           tf.keras.Model: returns a TensorFlow 2.20 model (compiled, untrained)
        """

        # Input validation
        self.__require_params(input_shape=input_shape)

        if (not gpu_device) or self.__GPU_count == 0:
            gpu_device = "/cpu:0"

        # Model hyperparameters and metadata
        model_name = "QnA Sample Model"
        opt = tf.keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        metrics = ["accuracy"]

        # Construct model & compile
        with tf.device(gpu_device):
            inp = tf.keras.layers.Input(input_shape, name="input_layer")
            inp_seq = inp[:, :, :, -1]
            X = MeanConcat()(inp)
            X = tf.expand_dims(X, axis=-1, name="expand_dims")
            X = AdapterPooler(386, shared_weights=True)(X)
            X = tf.reshape(X, (-1, X.shape[1], X.shape[2] * X.shape[3]))
            X = tf.concat([X, inp_seq], axis=2)
            X = tf.squeeze(X, axis=1)
            X = tf.keras.layers.Dense(2)(X)

            model = tf.keras.Model(inputs=inp, outputs=X, name=model_name)

        model.compile(loss=loss, optimizer=opt, metrics=metrics)

        # Print verbose output to console
        if verbose:
            self.__verbose_print(model, model_name, input_shape, opt, loss, metrics)

        return model
