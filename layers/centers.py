from __future__ import print_function
from utils import distance
from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer


class Centers(Layer):
    """Basic centers layer as in https://"""

    def __init__(self, units, T,
                 activation=None,
                 regularizer=None,#regularizers.l1_l2(l1=0.01, l2=0.01),
                 #initializer='uniform',
                 initializer=initializers.Orthogonal(gain=1.0, seed=None),
                 activity_regularizer=None,
                 constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Centers, self).__init__(**kwargs)
        self.units = units
        self.T = T
        self.centers = None
        self.activation = activations.get(activation)
        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.constraint = constraints.get(constraint)
        self.supports_masking = True

    def compute_output_shape(self, input_shapes):
        return tuple([input_shapes[0], self.units])

    def build(self, input_shapes):
        self.centers = self.add_weight(shape=(self.units,
                                              input_shapes[1]),
                                       initializer=self.initializer,
                                       name='centers',
                                       trainable=True,
                                       regularizer=self.regularizer,
                                       constraint=self.constraint)
        self.built = True

    def call(self, inputs, mask=None):
        output = -(distance(inputs, self.centers)) / self.T
        return self.activation(output)

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'initializer': initializers.serialize(
                      self.initializer),
                  'regularizer': regularizers.serialize(
                      self.regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'constraint': constraints.serialize(
                      self.constraint)
                  }

        base_config = super(Centers, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
