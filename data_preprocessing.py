import numpy as np
import pandas as pd
import tensorflow as tf 
import os
import cv2
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
import re


class preprocess_data:
    def __init__(self,image_path, caption_path,sample_size=30,size=(256,256), num_channels=3):
        self.image_path=image_path
        self.caption_path=caption_path
        self.sample_size=sample_size
        self.size=size
        self.num_channels=num_channels



    def preprocess_image_data(self):

        image_data=os.listdir(self.image_path)
        image_data=image_data[:self.sample_size]

        train=np.array([None]*self.sample_size)
        real_images=np.array([None]*self.sample_size)

        for j,i in enumerate(image_data):
            real_images[j] = np.array(plt.imread(os.path.join(self.image_path,i)))
            img=cv2.resize(real_images[j],self.size)
            train[j]=img.reshape(1, self.size[0], self.size[1],3)



        train=np.vstack(train)
        return image_data,train,real_images


    def preprocess_captions(self):
        train_images,_,_=self.preprocess_image_data()
        train_captions=pd.read_csv(self.caption_path,delimiter='|')
        train_captions.columns = ['image_name', 'comment_number', 'comment']

        def get_captions(train_images, train_captions):
            captions=[]
            for image_name in train_images:
                captions.append(train_captions[train_captions['image_name']==image_name]['comment'].iat[0])
            return captions


        def get_vocab(captions):
            m=captions.shape[0]
            sentence=[None]*m
            arr=[]
            
            for j,i in enumerate(captions):
                i=re.sub(' +', ' ',i)
                i=start_tag +' '+i+' '+end_tag
                sentence[j]=i.split()
                arr=arr+sentence[j]
        
            vocab_size=len(set(arr))
            
            fwd_dict={}
            rev_dict={}
    
            for ind, tokens in enumerate(set(arr)):
                fwd_dict[tokens]=ind
                rev_dict[ind]=tokens
                
            return sentence, vocab_size, fwd_dict, rev_dict



        captions=np.array(get_captions(train_images, train_captions))
        start_tag='<s>'
        end_tag='<e>'
        sentences,vocab_size, fwd_dict, rev_dict=get_vocab(captions)
        train_captions=[None]*len(sentences)

        for ind,sentence in enumerate(sentences):
            cap_array=None
            for token in sentence:
                row=[0]
                column=[fwd_dict[token]]
                data=[1]
                if cap_array==None:
                    cap_array=csr_matrix((data, (row, column)), shape=(1,vocab_size))
                    
                else:
                    cap_array=vstack((cap_array,csr_matrix((data, (row, column)), shape=(1,vocab_size))))
                    
            train_captions[ind]=cap_array


        return train_captions,vocab_size, (fwd_dict,rev_dict)



    def get_data(self):

        _,train_imgs,__=self.preprocess_image_data()
        train_captions,vocab_size,_=self.preprocess_captions()

        return train_imgs, train_captions,vocab_size




