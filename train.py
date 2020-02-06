from model import model
from data_preprocessing import preprocess_data
import tensorflow as tf
import argparse
from tqdm import tqdm

parser=argparse.ArgumentParser()

parser.add_argument('-img_path', type=str,default='C:\\Users\\Administrator\\Desktop\\flickr30k_images\\flickr30k_images')
parser.add_argument('-caption_path', type=str,default='C:\\Users\\Administrator\\Desktop\\flickr30k_images\\results.csv')
parser.add_argument('-learning_rate', type=float, default=0.0001)
parser.add_argument('-training_iters', type=int, default=5000)
parser.add_argument('-display_steps', type=int, default=10)
parser.add_argument('-bridge_size', type=int, default=1024)
parser.add_argument('-sample_size', type=int, default=30)
parser.add_argument('-size', type=tuple, default=(256,256))
parser.add_argument('-num_channels', type=int, default=3)

args = parser.parse_args()


class train_model(object):
    def __init__(self,args):
        self.training_iters=args.training_iters
        self.display_steps=args.display_steps

      


        processed_data=preprocess_data(image_path=args.img_path, caption_path=args.caption_path,sample_size=args.sample_size,size=args.size, num_channels=args.num_channels)
        self.train, self.train_captions,self.vocab_size=processed_data.get_data()

        self.x_caption=tf.placeholder(tf.float32,shape=[None, self.vocab_size],name='x_caption')
        self.x_inp=tf.placeholder(tf.float32, shape=[1, args.size[0], args.size[1], args.num_channels], name='x_input')
        self.y=tf.placeholder(tf.float32, shape=[None,self.vocab_size], name='y_image')
        mod=model(self.vocab_size,args.bridge_size,self.x_caption, self.x_inp, self.y,args.size)
        self.cost, self.optimizer, self.accuracy=mod.full_model(learning_rate=0.0001)


    def main(self):
        saver=tf.train.Saver()
        init=tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            m=len(self.train_captions)
            
            for epoch in tqdm(range(self.training_iters)):
                total_cst=0
                total_acc=0
                for i in tqdm(range(m)):
                
                    _, cst, acc=sess.run([self.optimizer,self.cost,self.accuracy],
                                        feed_dict={self.x_caption:self.train_captions[i][:-1].A,
                                                    self.x_inp:self.train[i:i+1],self.y:self.train_captions[i][1:].A})
                    total_cst+=cst
                    total_acc+=acc
                    
                if (epoch + 1) % self.display_steps == 0:
                    print('After ', (epoch + 1), 'iterations: Cost = ', total_cst / m, 'and Accuracy: ', total_acc * 100/ m , '%' )
                    save_path = saver.save(sess, "saved_model\model")        



if __name__=='__main__':
    model=train_model(args)
    model.main()