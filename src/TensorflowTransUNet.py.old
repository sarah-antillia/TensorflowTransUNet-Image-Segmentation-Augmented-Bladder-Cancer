# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# TensorflowTransUNet.py
# 2023/12/10 to-arai

# Some methods of TensorflowTransUNet class have been taken from the following web-sites.
# https://github.com/awsaf49/TransUNet-tf/blob/main/transunet/model.py
# 
#  MIT license

from __future__ import absolute_import

import os
import sys

import traceback

import numpy as np
from glob import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Input

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, concatenate

import sys


from ConfigParser import ConfigParser
from TensorflowUNet import TensorflowUNet

from losses import dice_coef, basnet_hybrid_loss, sensitivity, specificity
from losses import iou_coef, iou_loss, bce_iou_loss, bce_dice_loss

import transunet.encoder_layers as encoder_layers
import transunet.decoder_layers as decoder_layers
from transunet.resnet_v2 import  resnet_embeddings
#import tensorflow_addons as tfa
#import matplotlib.pyplot as plt
import transunet.utils as utils

import math

tfk = tf.keras
tfkl = tfk.layers
tfm = tf.math

MODELS_URL = 'https://storage.googleapis.com/vit_models/imagenet21k/'
        

MODEL = "model"
EVAL  = "eval"
INFER = "infer"

class TensorflowTransUNet(TensorflowUNet) :

  def __init__(self, config_file):
    #super().__init__(config_file)
    
    self.model_loaded = False

    self.config_file = config_file
    self.config = ConfigParser(config_file)
    self.show_history = self.config.get(MODEL, "show_history", dvalue=False)

    self.config.dump_all()
  
    num_classes     = self.config.get(MODEL, "num_classes")
    image_width     = self.config.get(MODEL, "image_width")
    image_height    = self.config.get(MODEL, "image_height")
    image_channels  = self.config.get(MODEL, "image_channels")
    base_filters    = self.config.get(MODEL, "base_filters")
    num_layers      = self.config.get(MODEL, "num_layers")

    self.patch_size       = self.config.get(MODEL, "patch_size", dvalue=16) 
    self.hybrid           = self.config.get(MODEL, "hybrid", dvalue=True)
    self.grid             = self.config.get(MODEL, "grid", dvalue=(14,14)) 
    self.hidden_size      = self.config.get(MODEL, "hidden_size", dvalue=612) #768,
    self.num_heads        = self.config.get(MODEL, "num_heads", dvalue=6)     #12,
    self.mlp_dim          = self.config.get(MODEL, "mlp_dim", dvalue=1280)    #3072,

    self.dropout_rate     = self.config.get(MODEL, "dropout_rate", dvalue=0.1)
    self.decoder_channels = self.config.get(MODEL, "decoder_channels", dvalue=[256,128,64,16])
    self.num_skip         = self.config.get(MODEL, "num_skip", dvalue=3)
    
    self.final_activation = self.config.get(MODEL, "final_activation", dvalue='sigmoid')
    self.pretrain         = self.config.get(MODEL, "pretrain", dvalue=False)
    self.freeze_enc_cnn   = self.config.get(MODEL, "freeze_enc_cnn", dvalue=True)

    self.model      = self.create(num_classes, image_height, image_width, image_channels,
                         base_filters = base_filters, num_layers = num_layers)
  
    learning_rate    = self.config.get(MODEL, "learning_rate")
    clipvalue        = self.config.get(MODEL, "clipvalue", dvalue=0.5)
    
    # Optimization
    # <---- !!! gradient clipping is important
    
    # 2024/03/05
    optimizer = self.config.get(MODEL, "optimizer", dvalue="Adam")
    if optimizer == "Adam":
      self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate,
         beta_1=0.9, 
         beta_2=0.999, 
         clipvalue=clipvalue,  #2023/06/26
         amsgrad=False)
      print("=== Optimizer Adam learning_rate {} clipvalue {} ".format(learning_rate, clipvalue))
    
    elif optimizer == "AdamW":
      # 2023/11/10  Adam -> AdamW (tensorflow 2.14.0~)
      self.optimizer = tf.keras.optimizers.AdamW(learning_rate = learning_rate,
         clipvalue=clipvalue,
         )
      print("=== Optimizer AdamW learning_rate {} clipvalue {} ".format(learning_rate, clipvalue))
        
    binary_crossentropy = tf.keras.metrics.binary_crossentropy
    binary_accuracy     = tf.keras.metrics.binary_accuracy
    
    # Default loss and metrics functions
    self.loss    = binary_crossentropy
    self.metrics = [binary_accuracy]
    
    # Read a loss function name from our config file, and eval it.
    # loss = "binary_crossentropy"
    self.loss  = eval(self.config.get(MODEL, "loss"))

    # Read a list of metrics function names, ant eval each of the list,
    # metrics = ["binary_accuracy"]
    metrics  = self.config.get(MODEL, "metrics")
    self.metrics = []
    for metric in metrics:
      self.metrics.append(eval(metric))
    
    print("--- loss    {}".format(self.loss))
    print("--- metrics {}".format(self.metrics))
    
  
    self.model.compile(optimizer= self.optimizer, loss =   self.loss, metrics= self.metrics)


  def load_pretrained(self, model, fname='R50+ViT-B_16.npz'):
    """Load model weights for a known configuration."""
    origin = MODELS_URL + fname
    print("=== load_pretrained {}".format(origin))
    local_filepath = tf.keras.utils.get_file(fname, origin, cache_subdir="weights")
    utils.load_weights_numpy(model, local_filepath)
    print("=== local filepath{}".format(local_filepath))

  def create(self, num_classes, image_height, image_width, image_channels,
               base_filters = 16, num_layers = 6):
    print("==== TensorflowTransUNet.create ")
    # Tranformer Encoder
    image_size = image_width
    
    assert image_size % self.patch_size == 0, "image_size must be a multiple of patch_size"
    x = tf.keras.layers.Input(shape=(image_size, image_size, 3))

    # Embedding
    if self.hybrid:
        print("--- hybrid {}".format(self.hybrid))
        grid_size = self.grid
        self.patch_size = image_size // 16 // grid_size[0]
        if self.patch_size == 0:
            self.patch_size = 1
        print("--- patch_size auto computed: patch_size = {}".format(self.patch_size))
        resnet50v2, features = resnet_embeddings(x, image_size=image_size, 
                                                 n_skip=self.num_skip, 
                                                 pretrain=self.pretrain)
        if self.freeze_enc_cnn:
            resnet50v2.trainable = False
        y = resnet50v2.get_layer("conv4_block6_preact_relu").output
        x = resnet50v2.input
    else:
        print("--- hybrid {}".format(self.hybrid))
        y = x
        features = None

    y = tfkl.Conv2D(
          filters=self.hidden_size,
          kernel_size=self.patch_size,
          strides=self.patch_size,
          padding="valid",
          name="embedding",
          trainable=True)(y)
    y = tfkl.Reshape(
          (y.shape[1] * y.shape[2], 
          self.hidden_size))(y)
    y = encoder_layers.AddPositionEmbs(
          name="Transformer/posembed_input", 
          trainable=True)(y)

    y = tfkl.Dropout(0.1)(y)

    # Transformer/Encoder
    for n in range(num_layers):
        
        y, _ = encoder_layers.TransformerBlock(
            n_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout_rate,
            name=f"Transformer/encoderblock_{n}",
            trainable=True)(y)
    y = tfkl.LayerNormalization(
        epsilon=1e-6, 
        name="Transformer/encoder_norm")(y)

    n_patch_sqrt = int(math.sqrt(y.shape[1]))

    y = tfkl.Reshape(
        target_shape=[n_patch_sqrt, n_patch_sqrt, self.hidden_size])(y)

    # Decoder CUP
    if len(self.decoder_channels):
        y = decoder_layers.DecoderCup(decoder_channels=self.decoder_channels, n_skip=self.num_skip)(y, features)

    # Segmentation Head
    y = decoder_layers.SegmentationHead(num_classes=num_classes, final_act=self.final_activation)(y)

    # Build Model
    model =  tfk.models.Model(inputs=x, outputs=y, name='TransUNet')
    
    # Load Pretrain Weights
    if self.pretrain:
        self.load_pretrained(model)
        
    return model
    

if __name__ == "__main__":
  try:
    config_file = "./train_eval_inf.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    if not os.path.exists(config_file):
      raise Exception("Not found " + config_file)
    model = TensorflowTransUNet(config_file)
    
  except:
    traceback.print_exc()
