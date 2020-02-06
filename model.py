from utils import model_utils
import tensorflow as tf


class model(object):
    def __init__(self, vocab_size,bridge_size,x_caption, x_inp, y,size):
        self.x_caption=x_caption
        self.x_inp=x_inp
        self.y=y
        self.size=size        
        self.bridge_size=bridge_size
        self.vocab_size=vocab_size
        self.wconv=tf.Variable(tf.truncated_normal([bridge_size, vocab_size],stddev=0.7))
        self.bconv=tf.Variable(tf.zeros([1,vocab_size]))
        self.Wi=tf.Variable(tf.truncated_normal([vocab_size,vocab_size], stddev=0.7))
        self.Wo=tf.Variable(tf.truncated_normal([vocab_size,vocab_size], stddev=0.7))
        self.Wf=tf.Variable(tf.truncated_normal([vocab_size,vocab_size], stddev=0.7))
        self.b=tf.Variable(tf.zeros([1,vocab_size]))
        self.mu=model_utils()

    def initialize_variables(self):
        return self.Wi, self.Wo,self.Wf, self.b, self.wconv, self.bconv


    def conv_net(self):

        layer_conv1=self.mu.conv_layer(inp=self.x_inp, kernel_shape=(3,3), num_kernels=32, num_channels=3 , suffix=1)
        
        layer_conv2=self.mu.conv_layer(inp=layer_conv1, kernel_shape=(3,3), num_kernels=32, num_channels=32 , suffix=2)
        
        maxpool1=tf.nn.max_pool(layer_conv2,ksize=[1, 2, 2, 1], strides=[1, 2, 2,1], padding= 'SAME')
        
        layer_conv3 = self.mu.conv_layer(inp=maxpool1, kernel_shape=(3, 3), num_kernels=64, num_channels=32, suffix='3')
        
        layer_conv4 = self.mu.conv_layer(inp=layer_conv3, kernel_shape=(3, 3), num_kernels=64, num_channels=64, suffix='4')
        
        maxpool2 = tf.nn.max_pool(layer_conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2,1], padding= 'SAME')
        
        layer_conv5 = self.mu.conv_layer(inp=maxpool2, kernel_shape=(3, 3), num_kernels=128, num_channels=64, suffix='5')
        
        layer_conv6 = self.mu.conv_layer(inp=layer_conv5, kernel_shape=(3, 3), num_kernels=128, num_channels=128, suffix='6')
        
        maxpool3 = tf.nn.max_pool(layer_conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2,1], padding= 'SAME')
        
        layer_conv7 =self.mu.conv_layer(inp=maxpool3, kernel_shape=(3, 3), num_kernels=256, num_channels=128, suffix='7')
        
        layer_conv8 = self.mu.conv_layer(inp=layer_conv7, kernel_shape=(3, 3), num_kernels=256, num_channels=256, suffix='8')

        flat_layer = self.mu.flatten_layer(layer_conv8, suffix='9')
        
        dense_layer_1 = self.mu.dense_layer(inputs=flat_layer, num_inputs=262144 , num_outputs=self.bridge_size, suffix='10')

        return dense_layer_1



    def full_model(self,learning_rate):
        hook=tf.slice(self.x_caption, [0,0],[1,self.vocab_size])
        h=self.conv_net()
        h,out=self.mu.rnn_unit(self.Wi,self.Wo,self.wconv,self.bconv,h,hook)

        def fn(prev, curr):
            h=prev[0]
            curr=tf.reshape(curr,[1,self.vocab_size])
            h, out=self.mu.rnn_unit(self.Wi ,self.Wo, self.Wf, self.b, h, curr)
            
            return h, out


        _, output=tf.scan(fn, self.x_caption[1:], initializer=(h,out))
        output=tf.squeeze(output, axis=1)
        outputs=tf.concat([out, output], axis=0)
        cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs,labels=self.y))

        optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)
        pred=tf.nn.softmax(outputs)
        correct_pred=tf.equal(tf.argmax(pred,1), tf.argmax(self.y,1))
        accuracy=tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        return cost, optimizer,accuracy




     

