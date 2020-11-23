#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf
import horovod.tensorflow.keras as hvd
import os, datetime, argparse, time
import numpy as np


# In[ ]:


parser = argparse.ArgumentParser(description='lab2', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model-name', default="I", help='model name, InceptionV3: I; ResNet50: R; VGG16: V')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--image-per-epoch', type=int, default=65536)
parser.add_argument('--scalability', default="strong", help='strong/weak')

args = parser.parse_args()


# In[ ]:


model_name = args.model_name
batch_size = args.batch_size
image_per_epoch = args.image_per_epoch
scalability = args.scalability


# In[ ]:


# Horovod: initialize Horovod.
hvd.init()
print("hvd.size: ", hvd.size())

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


# In[ ]:


print("scalability: ", scalability)

if scalability == "strong":
    global_batch_size = batch_size
    local_batch_size = int(global_batch_size / hvd.size())
elif scalability == "weak":
    global_batch_size = int(batch_size * hvd.size())
    local_batch_size = batch_size
    
batch_per_epoch = int(image_per_epoch / global_batch_size)

print("global_batch_size: ", global_batch_size)
print("local_batch_size: ", local_batch_size)
print("batch_per_epoch: ", batch_per_epoch)
print("", flush=True)


# In[ ]:


train_dir = "/users/PAS1777/brucechen/lab1/dataset/gender/train/"
val_dir = "/users/PAS1777/brucechen/lab1/dataset/gender/test/"
#train_dir = "/fs/scratch/PAS1777/imdb_wiki/dataset/gender/train/"
#val_dir = "/fs/scratch/PAS1777/imdb_wiki/dataset/gender/train/"
#train_dir = "/fs/scratch/PAS1777/tf_horovod/dataset/gender/train/"
#val_dir = "/fs/scratch/PAS1777/tf_horovod/dataset/gender/test/"


# In[ ]:


model_config = {"I": {"target_size": (299, 299), 
                      "preprocessing_function": tf.keras.applications.inception_v3.preprocess_input},
                "R": {"target_size": (224, 224), 
                      "preprocessing_function": tf.keras.applications.resnet.preprocess_input},
                "V": {"target_size": (224, 224), 
                      "preprocessing_function": tf.keras.applications.vgg19.preprocess_input},
               }


# In[ ]:


# Training data iterator.
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    width_shift_range=0.33, height_shift_range=0.33, zoom_range=0.5, horizontal_flip=True,
    preprocessing_function=model_config[model_name]["preprocessing_function"])
train_iter = train_gen.flow_from_directory(train_dir,
                                           batch_size=local_batch_size,
                                           target_size=model_config[model_name]["target_size"],
                                           class_mode='binary')
dataset = train_iter

# Validation data iterator.
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    zoom_range=(0.875, 0.875), preprocessing_function=model_config[model_name]["preprocessing_function"])
test_iter = test_gen.flow_from_directory(val_dir,
                                         batch_size=local_batch_size,
                                         target_size=model_config[model_name]["target_size"],
                                         class_mode='binary')
val_dataset = test_iter


# In[ ]:


if model_name == "I":
    model = tf.keras.applications.InceptionV3(weights=None, classes=2)
elif model_name == "R":
    model = tf.keras.applications.ResNet50(weights=None, classes=2)
elif model_name == "V":
    model = tf.keras.applications.VGG19(weights=None, classes=2)


# In[ ]:


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


# In[ ]:


# Horovod: adjust learning rate based on number of GPUs.
scaled_lr = 0.001 * hvd.size()
opt = tf.optimizers.Adam(scaled_lr)

# Horovod: add Horovod DistributedOptimizer.
opt = hvd.DistributedOptimizer(opt)

# Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
# uses hvd.DistributedOptimizer() to compute gradients.
model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                    optimizer=opt,
                    metrics=['accuracy'],
                    experimental_run_tf_function=False)


# In[ ]:


time_callback = TimeHistory()

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, initial_lr=scaled_lr, verbose=1),
    time_callback,
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
#if hvd.rank() == 0:
    #callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))


# In[ ]:


# Horovod: write logs on worker 0.
verbose = 1 if hvd.rank() == 0 else 0

# Train the model.
# Horovod: adjust number of steps based on number of GPUs.
model.fit(dataset, steps_per_epoch=batch_per_epoch, callbacks=callbacks, epochs=5, verbose=verbose)


# In[ ]:


print(np.mean(time_callback.times[3:-1]))

