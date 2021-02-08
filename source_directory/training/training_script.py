import tensorflow as tf
import horovod.tensorflow.keras as hvd

import argparse
import boto3
import logging
import math
import numpy as np
import os
import random
import time

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras import backend as K
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam


logging.getLogger().setLevel(logging.INFO)
        

class Sync2S3(Callback):
    def __init__(self, log_dir, s3log_dir):
        super(Sync2S3, self).__init__()
        self.log_dir = log_dir
        self.s3log_dir = s3log_dir
    
    def on_epoch_end(self, batch, logs={}):
        os.system('aws s3 sync ' + self.log_dir + ' ' + self.s3log_dir)


def create_model():
    
    # create model with resnet base
    resnet_base = ResNet50V2(include_top=False, weights="imagenet", input_shape=(32,32,3), pooling='avg',)
    for layer in resnet_base.layers:
        layer.trainable = True

    model_input = Input(shape=(32,32,3), name='image_input')
    model = resnet_base(model_input)
    model = Dense(256)(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Dropout(rate=0.3)(model)
    model = Dense(10)(model)
    outputs = Activation('softmax')(model)

    model = Model(inputs=model_input, outputs=outputs)
    
    return model


def load_dataset(epochs, batch_size, channel_name):

    # load from with tf.data API
    if args.tfdata_s3uri:
        s3uri = '{}/{}'.format(args.tfdata_s3uri, channel_name)
        bucket = s3uri.split('/')[2]
        prefix = os.path.join(*s3uri.split('/')[3:])
        s3_client = boto3.client('s3')
        objects_list = s3_client.list_objects(Bucket=bucket, Prefix=prefix)
        files = []
        for obj in objects_list['Contents']:
            files.append('s3://{}/{}'.format(bucket, obj['Key']))
        if args.use_horovod:
            files, smallest_amount_samples = get_files_for_processor(files)
        print("Files to be read from {} set:".format(channel_name))
        for f in files:
            print(f)
    else:
        pass
        
    dataset = tf.data.TFRecordDataset(files)

    # compute number of batches per epoch
    if args.use_horovod:
        num_batches_per_epoch = math.floor(smallest_amount_samples/batch_size)
    else:
        num_samples = sum(1 for _ in dataset)
        print("{} set has {} samples.".format(channel_name, num_samples))
        num_batches_per_epoch = math.floor(num_samples/batch_size)

    # parse records
    dataset = dataset.map(_dataset_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # shuffle records for training set
    if channel_name == 'train':
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.map(image_augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # batch the dataset
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # repeat and prefetch
    dataset = dataset.repeat(epochs)
    dataset = dataset.prefetch(1000)
    
    return dataset, num_batches_per_epoch


def get_files_for_processor(files):
    split_files = np.array_split(files, hvd.size())
    local_files = split_files[hvd.rank()].tolist()
    
    # get smallest amount of samples against all GPUs/processors
    for i, files in enumerate(split_files):
        dataset = tf.data.TFRecordDataset(files.tolist())
        if i == 0:
            smallest_amount_samples = sum(1 for _ in dataset)
        else:
            smallest_amount_samples = min(smallest_amount_samples, sum(1 for _ in dataset))
        if hvd.rank() == 0:
            print("Dataset {} has {} samples.".format(i, sum(1 for _ in dataset)))
    
    if hvd.rank() == 0:
        print("Smallest amount of samples is {}.".format(smallest_amount_samples))
    return local_files, smallest_amount_samples


def image_augmentation(image, label):
    image = image['image_input']
    
    # Add 6 pixels of padding
    image = tf.image.resize_with_crop_or_pad(image, 32 + 6, 32 + 6) 
    # Random crop back to the original size
    image = tf.image.random_crop(image, size=[32, 32, 3])
    
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    image = tf.image.rot90(image, k=np.random.randint(4))
    image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
    
    image = {'image_input': image}
    return image, label
    
    
def _dataset_parser(value):
    
    # create a dictionary describing the features    
    sample_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    # parse to tf
    example = tf.io.parse_single_example(value, sample_feature_description)
    
    # decode from bytes to tf types
    # NOTE: example key must match the name of the Input layer in the keras model
    example['image'] = tf.io.decode_raw(example['image'], tf.uint8)
    example['image'] = tf.reshape(example['image'], (32,32,3))
    
    # preprocess for resnset
    # see https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet_v2/preprocess_input
    example['image'] = tf.cast(example['image'], tf.float32)
    example['image'] = tf.keras.applications.resnet_v2.preprocess_input(example['image'])
    
    # parse for input to neural network and loss function
    sample_data = {'image_input': example['image']}

    label = tf.cast(example['label'], tf.int32)
    label = tf.one_hot(indices=label, depth=10)
    
    return sample_data, label


def save_model(model, model_output_dir):
    
    # https://www.tensorflow.org/guide/keras/save_and_serialize
    logging.info("Saving model used model.save()")

    tf.saved_model.save(model, os.path.join(model_output_dir, 'cinic10_classifier', '1'))

    logging.info("Model successfully saved to: {}".format(model_output_dir))

    return


def main(args):
    
    if args.use_horovod:
        ## set up horovod for distributed training (multiple instances with multi-gpu)
        hvd.init()
        size = hvd.size()
        print("Horovod size:", size)
        print("Local horovod rank:", hvd.local_rank())
        print("Global horovod rank:", hvd.rank())

        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        
    else:
        ## set up replicas for multiple gpus
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


    ## create and compile the model
    print("Creating model")
    if args.use_horovod:
        model = create_model()
        distributed_learning_rate = size*args.learning_rate
        optimizer = Adam(lr=distributed_learning_rate, decay=args.weight_decay)
        optimizer = hvd.DistributedOptimizer(optimizer)
        print("Compiling model")
        model.compile(loss=CategoricalCrossentropy(),
                      optimizer=optimizer,
                      experimental_run_tf_function=False,
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])
    
    else:
        with strategy.scope():
            model = create_model()
            optimizer = Adam(lr=args.learning_rate, decay=args.weight_decay)

            ## compile model
            print("Compiling model")
            model.compile(loss=CategoricalCrossentropy(),
                          optimizer=optimizer,
                          experimental_run_tf_function=False,
                          metrics=[tf.keras.metrics.CategoricalAccuracy()],
                         )
    
    
    ## set up callbacks
    logging.info("Setting callbacks")
    tfLearningRatePlateau = tf.keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1)
    log_dir = './tf_logs/'
    verbose = 0
    if args.use_horovod:
        callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
            tfLearningRatePlateau,
        ]

        if hvd.rank() == 0:
            callbacks.append(TensorBoard(log_dir=log_dir))
            callbacks.append(Sync2S3(log_dir=log_dir, s3log_dir=args.tensorboard_logs_s3uri))
            verbose = 2

    else:
        callbacks = [
            tfLearningRatePlateau,
            TensorBoard(log_dir=log_dir),
            Sync2S3(log_dir=log_dir, s3log_dir=args.tensorboard_logs_s3uri),
        ]
        verbose = 2

        
    ## load the datasets
    print("Loading datasets")
    train_dataset, num_train_batches_per_epoch = load_dataset(
        args.epochs, args.batch_size, 'train')
    validation_dataset, num_validation_batches_per_epoch = load_dataset(
        args.epochs, args.batch_size, 'validation')
    
    
    ## start training
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit    
    print("Starting training")
    model.fit(x=train_dataset,
              steps_per_epoch=num_train_batches_per_epoch,
              epochs=args.epochs,
              validation_data=validation_dataset,
              validation_steps=num_validation_batches_per_epoch,
              verbose=2,
              callbacks=callbacks,
             )

    
    ## save model
    if args.use_horovod:
        if hvd.rank()==0:
            save_model(model, args.model_output_dir)
    else:
        save_model(model, args.model_output_dir)
        
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=False, default=os.environ.get('SM_CHANNEL_TRAIN'),
                        help='The directory where training input data is stored.')
    parser.add_argument('--validation', type=str, required=False, default=os.environ.get('SM_CHANNEL_VALIDATION'),
                        help='The directory where validation input data is stored.')
    parser.add_argument('--tfdata-s3uri', type=str, required=False,
                        help='The base S3 URI to tfrecords when using tf.data API.')
    parser.add_argument('--use-horovod', type=int, required=True, choices=[0,1],
                        help='Whether or not to use distributed training with Horovod.')
    parser.add_argument('--model_dir', type=str, required=False,
                        help="""\
                        The S3 path where the model will be uploaded upon training completion.
                        SageMaker will automatically pass this to the script.\
                        """                      )
    parser.add_argument('--model_output_dir', type=str, default=os.environ.get('SM_MODEL_DIR'),
                        help="""\
                        A string that represents the local path where the training job writes the model artifacts to.
                        After training, artifacts in this directory are uploaded to S3 for model hosting.
                        This is different than the model_dir argument passed in your training script,
                        which can be an S3 location. SM_MODEL_DIR is always set to /opt/ml/model.\
                        """)
    parser.add_argument('--tensorboard-logs-s3uri', type=str, required=True,
                        help='S3 URI to store tensorboard logs')
    parser.add_argument('--epochs', type=int, default=10,
                        help='The number of steps to use for training.')
    parser.add_argument('--batch-size',type=int, default=32,
                        help='Batch size for training.')
    parser.add_argument('--weight-decay', type=float, default=2e-4,
                        help='Weight decay for convolutions.')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help="""\
                        This is the inital learning rate value. The learning rate will decrease
                        during training.\
                        """)
    args = parser.parse_args()
    main(args)