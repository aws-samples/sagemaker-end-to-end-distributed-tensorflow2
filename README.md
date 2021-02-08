Sharding, distributed training and inference with TensorFlow 2 on SageMaker
=====================================================================================
This code repository contains an example of a full end-to-end machine learning cycle that includes code to prepare the data in shards with TFRecords, to train on multiple instances (distributed training) and to deploy the resulting model for use with real-time API calls and offline batch jobs with TFServing. 

Data preparation
----------------
We begin by downloading the [CINIC-10 dataset](https://datashare.is.ed.ac.uk/handle/10283/3192), an extension of CIFAR-10 (60k images, 32x32 RGB pixels) with a subset of ImageNet (210k images, downsampled to 32x32 RGB). Each image belongs to one of ten classes, providing us with a sufficiently large dataset to demonstrate the application of distributed training on a multi-class classification problem.

The data is then wrapped up into a series of [TFRecord files](https://www.tensorflow.org/tutorials/load_data/tfrecord) to optimize data loading through techniques such as pre-fetching to prevent GPU starvation. We choose to shard the images into a number of files that is some multiple of 8 since we will train on 8 separate GPUs.

Model training
--------------
There are two calls to create a training job on [Amazon SageMaker](https://sagemaker.readthedocs.io/en/stable/) in this repo to demonstrate the code change that's required to switch from training on a single instance with 8 GPUs to training on two separate instances with 4 GPUs each. In the former case we use [TensorFlow's built-in MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy) API, while in the latter [Horovod for Keras](https://horovod.readthedocs.io/en/stable/keras.html) is utilized.

Inference
---------
Before setting up the inference pipeline, we pull the model from S3 in the location specified in the SageMaker training job and evaluate on the test set to confirm it's performance. The model is then deployed as an endpoint for real-time use, and separately as a Batch Transform job. A custom inference script is provided for consuming images passed over an API call, or as a TFRecord dataset.

Running the code
----------------
To run this demo, enter the `notebooks` folder and execute each cell in the sequence that the notebooks are named:

1. `01-prep_dataset.ipynb`
2. `02-launch_training_job.ipynb`
3. `03-inference_pileline.ipynb`

The training and inference script are located under the `source_directory` folder.

License
-------
This sample code is made available under the MIT-0 license. See the LICENSE file.