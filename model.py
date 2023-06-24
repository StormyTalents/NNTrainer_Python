import numpy as np
import tensorflow as tf
import math

class Model(object):
    def __init__(self, input_shape, output_shape, components):
        self.input_shape = input_shape and \
            (input_shape if (type(input_shape) is list or type(input_shape) is tuple) \
             else [input_shape])
        self.output_shape = output_shape and \
            (output_shape if (type(output_shape) is list or type(output_shape) is tuple) \
             else [output_shape])
        self.components = components
        self.component_map = {component.name: component for component in components
                                if component.name is not None}
        self.link()
        try:
            self.configure()
        except Exception:
            print '\n', self, '\n'
            raise
        
    def configure(self):
        # Single backwards pass through linear dependencies
        inferred_output_indices = set()
        inferred_input_shape = None
        given_input_shape = self.output_shape
        for i in reversed(range(len(self.components))):
            assert self.components[i].output_shape is None or \
                (inferred_input_shape or given_input_shape) is None, \
                'output_shape of ' + type(self.components[i]).__name__ + ' at index ' + str(i) + \
                    ' can be inferred, and should not be specified'
            self.components[i].output_shape = self.components[i].output_shape or \
                inferred_input_shape or given_input_shape
            if self.components[i].output_shape:
                inferred_output_indices.add(i)
            given_input_shape = self.components[i].input_shape
            inferred_input_shape = self.components[i].output_shape and \
                self.components[i]._infer_input_shape(self, self.components[i].output_shape)
            if i == 0 and inferred_input_shape:
                assert given_input_shape == inferred_input_shape, \
                    'Model input and output shapes are incompatible given the specified components'
            self.components[i].direct_input_shape = given_input_shape or inferred_input_shape
                                      
        # Multiple forward passes through the linear dependencies to resolve recurrent inputs
        last = None
        while True:
            fingerprint = self.fingerprint()
            if last == fingerprint:
                break
            last = fingerprint
            inferred_output_shape = None
            given_output_shape = self.input_shape
            for i in range(len(self.components)):
                self.components[i].direct_input_shape = self.components[i].direct_input_shape or \
                    inferred_output_shape or given_output_shape
                given_output_shape = self.components[i].output_shape
                inferred_output_shape = self.components[i].input_shape and \
                    self.components[i]._infer_output_shape(self, self.components[i].input_shape)
                if i < len(self.components) - 1:
                    assert given_output_shape is None or inferred_output_shape is None or \
                        i in inferred_output_indices, \
                        'output_shape of ' + type(self.components[i]).__name__ + \
                            ' at index ' + str(i) + ' can be inferred, and should not be specified'
                if i == len(self.components) - 1 and inferred_output_shape:
                    assert given_output_shape == inferred_output_shape, \
                        'Model input and output shapes are incompatible given the specified components'
                self.components[i].output_shape = given_output_shape or inferred_output_shape
                if self.components[i].output_shape:
                    inferred_output_indices.add(i)
                                    
        # Explore each dependency tree for reccurent dependencies + inference check on output sizes
        # DFS expansion ensures that if an issue is found, it's found at the root of the problem
        component_index_map = {self.components[i]: i for i in range(len(self.components))}
        visited = set()
        stack = self.components[::-1]
        while len(stack):
            component = stack[-1]
            new = False
            for recurrent in component.recurrent_inputs:
                if recurrent not in visited:
                    stack.append(recurrent)
                    new = True
            if new: continue
            component = stack.pop()
            if component not in visited:
                assert component.output_shape, 'output_shape of ' + \
                    type(component).__name__ + ' at index ' + str(component_index_map[component]) + \
                    ' cannot be inferred because the model is ambiguous' + \
                    ' (likely due to ambiguous output sizes or variable-size recurrent inputs).' + \
                    ' Please specify the output_shape or modify your network.'
                inferred_output_shape = component._infer_output_shape(self, component.input_shape)
                assert inferred_output_shape is None or \
                    component.output_shape == inferred_output_shape, \
                    'Model input and output shapes are incompatible given the specified components'
                visited.add(component)
                
        for component in self.components:
            component.input_size = np.prod(component.input_shape)
            component.output_size = np.prod(component.output_shape)
            component.configured = True
        
    def link(self):
        for component in self.components:
            component.link(self)
    
    def get_component_by_name(self, name):
        return self.component_map.get(name, None)
    
    def fingerprint(self):
        return tuple(map(lambda c: (c.input_shape, c.output_shape), self.components))
        
    def __repr__(self):
        return '\n'.join(map(str, self.components))
    

class RecurrentModel(Model):
    def __init__(self, input_shape, output_shape, components):
        Model.__init__(self, input_shape, output_shape, components)
    
    def build(self):
        raise NotImplementedError('Recurrent model building is not yet implemented')
        x = tf.placeholder(tf.float32, [None] + self.input_shape)
        out = x
#        for component in self.components:
#            component.initialize()
#            out = component.apply(out)
        return x, out
    

class FeedForwardModel(Model):
    def __init__(self, input_shape, output_shape, components):
        Model.__init__(self, input_shape, output_shape, components)
                
    def build(self):
        x = tf.placeholder(tf.float32, [None] + self.input_shape)
        out = x
        for component in self.components:
            component.initialize()
            out = component.apply(out)
        return x, out
        

class Component(object):
    def __init__(self, params={}, name=None, recurrent_inputs=[], **kwargs):
        self.name = name
        self.recurrent_inputs = recurrent_inputs
        self.params = params
        self.kwargs = kwargs
        self.direct_input_shape = None
        self.output_shape = self.output_shape if hasattr(self, 'output_shape') else None
        self.output_shape = self.output_shape and \
            (self.output_shape if type(self.output_shape) is list else [self.output_shape])
        self.cached = None
        self.configured = False
            
    def link(self, model):
        self.recurrent_inputs = map(lambda r: model.get_component_by_name(r), self.recurrent_inputs)
    
    def initialize(self):
        pass
    
        
    def recurrents_well_defined(self):
        return all(map(lambda r: r.output_shape is not None, self.recurrent_inputs))
    
    def get_input_shape(self):
        if self.configured and self.cached is not None:
            return self.cached
        recurrent_shapes = map(lambda r: r.output_shape, self.recurrent_inputs)
        if self.direct_input_shape is None or None in recurrent_shapes:
            return None
        recurrent_shapes = map(lambda r: r.output_shape, self.recurrent_inputs)
        assert all(map(lambda s: s[1:] == self.direct_input_shape[1:], recurrent_shapes)), \
            'Recurrent shape cannot be concatenated onto input shape'
        result = [self.direct_input_shape[0] + sum(map(lambda s: s[0], recurrent_shapes))] + \
            self.direct_input_shape[1:]
        if self.configured:
            self.cached = result
        return result
    
    def illegal_operation(self, val):
        raise Exception('Input shape cannot be set')
    
    input_shape = property(get_input_shape, illegal_operation)
        
    def _infer_output_shape(self, model, input_shape):
        return self.infer_output_shape(input_shape) if self.recurrents_well_defined() else None
    def _infer_input_shape(self, model, output_shape):
        return self.infer_input_shape(output_shape) if self.recurrents_well_defined() else None
        
    def infer_output_shape(self, input_shape):
        return None
    def infer_input_shape(self, output_shape):
        return None
    
    def __repr__(self):
        recurrent_shapes = map(lambda r: r.output_shape, self.recurrent_inputs)
        return (type(self).__name__ + '(' + ('name=\'' + self.name + '\', ' if self.name else '') + \
            'input_shape=' + str(self.direct_input_shape) + \
            (('+' + '+'.join(map(lambda r: str(r.output_shape), self.recurrent_inputs)) + \
              '=' + ('???' if self.direct_input_shape and None not in recurrent_shapes and \
                     not all(map(lambda s: s[1:] == self.direct_input_shape[1:], recurrent_shapes)) \
                     else str(self.input_shape)))
                if len(self.recurrent_inputs) else '') + \
            ', output_shape=' + str(self.output_shape) + \
            ((', recurrent_inputs=' + str(map(lambda r: r.name, self.recurrent_inputs)))
                if len(self.recurrent_inputs) else '') + \
            ', '.join([''] + map(lambda k: k + '=' + str(self.params[k]), self.params)) + \
            ', '.join([''] + map(lambda k: k + '=' + str(self.kwargs[k]), self.kwargs)) + \
            ')').replace('None', '???')
    
    
class FullyConnectedComponent(Component):
    def __init__(self, output_shape=None, **kwargs):
        self.output_shape = output_shape
        Component.__init__(self, **kwargs)
    
    def initialize(self):
        self.weights = tf.Variable(
            tf.random_normal([self.input_size, self.output_size],
                             stddev=np.sqrt(2./self.input_size)))
        self.biases = tf.Variable(tf.zeros(self.output_size))
        
    def apply(self, x):
        x = tf.reshape(x, [-1, self.input_size])
        x = tf.add(tf.matmul(x, self.weights), self.biases)
        x = tf.reshape(x, [-1] + self.output_shape)
        return x
    
    
class Conv1DComponent(Component):
    def __init__(self, filter_width,
                 stride=1, padding='SAME', num_kernels=1, **kwargs):
        self.filter_width = filter_width
        self.stride = stride
        self.padding = padding
        self.num_kernels = num_kernels
        Component.__init__(self, {'filter_width': filter_width,
                                  'stride': stride,
                                  'padding': padding,
                                  'num_kernels': num_kernels},
                           **kwargs)
    
    def initialize(self):
        in_channels = self.input_shape[1] if len(self.input_shape) == 2 else 1
        self.weights = tf.Variable(
            tf.random_normal([self.filter_width, in_channels, self.num_kernels],
                             stddev=np.sqrt(2./(self.filter_width*in_channels*self.num_kernels))))
        self.biases = tf.Variable(tf.zeros(self.num_kernels))
        
    # Calculate output as a function of stride and padding type
    def infer_output_shape(self, input_shape):
        assert len(input_shape) >= 1 and len(input_shape) <= 2, \
            'input_shape of ' + type(self).__name__ + ' must be 1D or 2D (channel dimension)'
        if self.padding == 'SAME':
            return [int(math.ceil(input_shape[0] / float(self.stride)))] + \
                ([self.num_kernels] if self.num_kernels > 1 else [])
        else:
            return [int(math.ceil((input_shape[0] - self.filter_width + 1)/float(self.stride)))] + \
                ([self.num_kernels] if self.num_kernels > 1 else [])
        
    def apply(self, x):
        includes_channel_dimension = len(self.input_shape) == 2
        if not includes_channel_dimension:
            x = tf.expand_dims(x, 2)
        x = tf.nn.conv1d(x, self.weights, stride=self.stride, padding=self.padding,
                         **self.kwargs)
        x = tf.nn.bias_add(x, self.biases)
        if not includes_channel_dimension and self.num_kernels == 1:
            x = tf.squeeze(x, 2)
        return x
    
    
class Conv2DComponent(Component):
    def __init__(self, filter_size,
                 stride=None, strides=[1, 1], padding='SAME', num_kernels=1, **kwargs):
        if type(filter_size) is not list and type(filter_size) is not tuple:
            filter_size = [filter_size] * 2
        self.filter_width = filter_size[0]
        self.filter_height = filter_size[1]
        if stride is not None:
            strides = [stride] * 2
        assert len(strides) == 2
        self.strides = strides
        self.padding = padding
        self.num_kernels = num_kernels
        Component.__init__(self, {'filter_size': filter_size,
                                  'strides': strides,
                                  'padding': padding,
                                  'num_kernels': num_kernels},
                           **kwargs)
    
    def initialize(self):
        in_channels = self.input_shape[2] if len(self.input_shape) == 3 else 1
        self.weights = tf.Variable(
            tf.random_normal(
                [self.filter_width, self.filter_height, in_channels, self.num_kernels],
                stddev=np.sqrt(
                    2./(self.filter_width*self.filter_height*in_channels*self.num_kernels))))
        self.biases = tf.Variable(tf.zeros(self.num_kernels))
        
    # Calculate output as a function of strides and padding type
    def infer_output_shape(self, input_shape):
        assert len(input_shape) >= 2 and len(input_shape) <= 3, \
            'input_shape of ' + type(self).__name__ + ' must be 2D or 3D (channel dimension)'
        if self.padding == 'SAME':
            return [int(math.ceil(input_shape[0] / float(self.strides[0]))),
                    int(math.ceil(input_shape[1] / float(self.strides[1])))] + \
                ([self.num_kernels] if self.num_kernels > 1 else [])
        else:
            return [int(math.ceil(
                        (input_shape[0] - self.filter_width + 1)/float(self.strides[0]))),
                    int(math.ceil(
                        (input_shape[1] - self.filter_height + 1)/float(self.strides[1])))] + \
                    ([self.num_kernels] if self.num_kernels > 1 else [])
        
    def apply(self, x):
        includes_channel_dimension = len(self.input_shape) == 3
        if not includes_channel_dimension:
            x = tf.expand_dims(x, 3)
        x = tf.nn.conv2d(x, self.weights,
                         strides=([1]+self.strides+[1]), padding=self.padding,
                         **self.kwargs)
        x = tf.nn.bias_add(x, self.biases)
        if not includes_channel_dimension and self.num_kernels == 1:
            x = tf.squeeze(x, 3)
        return x
    
    
class Pool1DComponent(Component):
    def __init__(self, pool_fn, window_width,
                 stride=None, padding='SAME', **kwargs):
        self.pool_fn = pool_fn
        if stride is None:
            stride = window_width
        self.window_width = window_width
        self.stride = stride
        self.padding = padding
        Component.__init__(self, {'window_width': window_width,
                                  'stride': stride,
                                  'padding': padding},
                           **kwargs)
    
    def initialize(self):
        pass
        
    # Calculate output as a function of strides and padding type
    def infer_output_shape(self, input_shape):
        assert len(input_shape) >= 1 and len(input_shape) <= 2, \
            'input_shape of ' + type(self).__name__ + ' must be 1D or 2D (channel dimension)'
        if self.padding == 'SAME':
            return [int(math.ceil(input_shape[0] / float(self.stride)))] + \
                ([input_shape[1]] if len(input_shape) == 2 else [])
        else:
            return [int(math.ceil(
                        (input_shape[0] - self.window_width + 1)/float(self.stride)))] + \
                    ([input_shape[1]] if len(input_shape) == 2 else [])
        
    def apply(self, x):
        includes_channel_dimension = len(self.input_shape) == 2
        if not includes_channel_dimension:
            x = tf.expand_dims(x, 2)
        x = tf.expand_dims(x, 2)
        x = self.pool_fn(x, ksize=([1, self.window_width, 1, 1]),
                         strides=([1, self.stride, 1, 1]), padding=self.padding,
                         **self.kwargs)
        x = tf.squeeze(x, 2)
        if not includes_channel_dimension:
            x = tf.squeeze(x, 2)
        return x
    
    
class Pool2DComponent(Component):
    def __init__(self, pool_fn, window_size,
                 stride=None, strides=None, padding='SAME', **kwargs):
        self.pool_fn = pool_fn
        if type(window_size) is not list and type(window_size) is not tuple:
            window_size = [window_size] * 2
        if stride is not None:
            strides = [stride] * 2
        if strides is None:
            strides = window_size
        self.window_size = window_size
        assert len(strides) == 2
        self.strides = strides
        self.padding = padding
        Component.__init__(self, {'window_size': window_size,
                                  'strides': strides,
                                  'padding': padding},
                           **kwargs)
            
    # Calculate output as a function of strides and padding type
    def infer_output_shape(self, input_shape):
        assert len(input_shape) >= 2 and len(input_shape) <= 3, \
            'input_shape of ' + type(self).__name__ + ' must be 2D or 3D (channel dimension)'
        if self.padding == 'SAME':
            return [int(math.ceil(input_shape[0] / float(self.strides[0]))),
                    int(math.ceil(input_shape[1] / float(self.strides[1])))] + \
                ([input_shape[2]] if len(input_shape) == 3 else [])
        else:
            return [int(math.ceil(
                        (input_shape[0] - self.window_size[0] + 1)/float(self.strides[0]))),
                    int(math.ceil(
                        (input_shape[1] - self.window_size[1] + 1)/float(self.strides[1])))] + \
                    ([input_shape[2]] if len(input_shape) == 3 else [])
        
    def apply(self, x):
        includes_channel_dimension = len(self.input_shape) == 3
        if not includes_channel_dimension:
            x = tf.expand_dims(x, 3)
        x = self.pool_fn(x, ksize=([1]+self.window_size+[1]),
                         strides=([1]+self.strides+[1]), padding=self.padding,
                         **self.kwargs)
        if not includes_channel_dimension:
            x = tf.squeeze(x, 3)
        return x
    
    
class AvgPool1DComponent(Pool1DComponent):
    def __init__(self, window_width,
                 stride=None, padding='SAME', **kwargs):
        Pool1DComponent.__init__(self, tf.nn.avg_pool, window_width,
                                 stride=stride, padding=padding, **kwargs)
    
    
class AvgPool2DComponent(Pool2DComponent):
    def __init__(self, window_size,
                 stride=None, strides=None, padding='SAME', **kwargs):
        Pool2DComponent.__init__(self, tf.nn.avg_pool, window_size,
                                 stride=stride, strides=strides, padding=padding, **kwargs)
    
    
class MaxPool1DComponent(Pool1DComponent):
    def __init__(self, window_width,
                 stride=None, padding='SAME', **kwargs):
        Pool1DComponent.__init__(self, tf.nn.max_pool, window_width,
                                 stride=stride, padding=padding, **kwargs)
    
    
class MaxPool2DComponent(Pool2DComponent):
    def __init__(self, window_size,
                 stride=None, strides=None, padding='SAME', **kwargs):
        Pool2DComponent.__init__(self, tf.nn.max_pool, window_size,
                                 stride=stride, strides=strides, padding=padding, **kwargs)
    
    
class DropoutComponent(Component):
    def __init__(self, keep_prob, **kwargs):
        self.keep_prob = keep_prob
        Component.__init__(self, {'keep_prob': keep_prob},
                           **kwargs)
        
    # Identical input and output shapes
    def infer_output_shape(self, input_shape):
        return input_shape
    def infer_input_shape(self, output_shape):
        return output_shape
        
    def apply(self, x):
        x = tf.nn.dropout(x, self.keep_prob, **self.kwargs)
        return x
    
    
class ReshapeComponent(Component):
    def __init__(self, output_shape, **kwargs):
        self.output_shape = output_shape
        Component.__init__(self, **kwargs)
        
    def initialize(self):
        assert self.input_size == self.output_size, 'Cannot reshape ' + \
            str(self.input_shape) + ' to ' + str(self.input_shape)
        
    def apply(self, x):
        x = tf.reshape(x, [-1] + self.output_shape, **self.kwargs)
        return x
    
    
# Could be implemented using a CustomComponent, but kept as an alias
class ActivationComponent(Component):
    def __init__(self, activation_fn, **kwargs):
        self.activation_fn = activation_fn
        Component.__init__(self, {'activation_fn': activation_fn.__name__},
                           **kwargs)
        
    # Activation functions have identical input and output shapes
    def infer_output_shape(self, input_shape):
        return input_shape
    def infer_input_shape(self, output_shape):
        return output_shape
        
    def apply(self, x):
        x = self.activation_fn(x)
        return x
    
    
# CustomComponents must retain the shape of the input
class CustomComponent(Component):
    def __init__(self, fn, **kwargs):
        self.fn = fn
        Component.__init__(self, **kwargs)
        
    # Must have identical input and output shapes, else inference is impossible
    def infer_output_shape(self, input_shape):
        return input_shape
    def infer_input_shape(self, output_shape):
        return output_shape
        
    def apply(self, x):
        x = self.fn(x, **self.kwargs)
        return x
    

