import tensorflow as tf 


class model_utils:

    def create_weights(self,shape, suffix):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.7,name=f'W_{suffix}'))

    def create_biases(self,size, suffix):
        return tf.Variable(tf.zeros([size],name=f'b_{suffix}'))


    def conv_layer(self,inp, kernel_shape, num_channels, num_kernels, suffix):
        filter_shape=[kernel_shape[0], kernel_shape[1], num_channels, num_kernels]
        weights=self.create_weights(shape=filter_shape, suffix=suffix)
        biases=self.create_biases(num_kernels, suffix)
        layer=tf.nn.conv2d(input=inp, filter=weights, padding='SAME', strides=[1,1,1,1], name=f'conv_{suffix}')
        layer+=biases
        layer=tf.nn.relu6(layer, name=f'relu_{suffix}')
        return layer


    def flatten_layer(self,layer, suffix):
        layer_shape=layer.get_shape()
        features=layer_shape[1:4].num_elements()
        layer=tf.reshape(layer, [-1, features], name=f'flatten_{suffix}')
        return layer


    def dense_layer(self,inputs,num_inputs, num_outputs, suffix,use_relu=True,):
        weights=self.create_weights([num_inputs, num_outputs], suffix)
        biases=self.create_biases(num_outputs,suffix)
        
        layer=tf.matmul(inputs, weights)+biases
        layer=tf.nn.relu(layer)
        return layer

    def rnn_unit(self,win, wout, wfwd, b, hprev, inp):
        h=tf.tanh(tf.add(tf.add(tf.matmul(inp,win), tf.matmul(hprev,wfwd)), b))
        out=tf.matmul(h,wout)
        return h, out


    