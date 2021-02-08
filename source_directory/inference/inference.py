# for additional details on payload handling, please see:
# https://github.com/aws/sagemaker-tensorflow-serving-container#pre/post-processing

import base64
from google.protobuf.json_format import MessageToDict
from io import BytesIO
import json
import numpy as np
import os
from PIL import Image
import requests
import tensorflow as tf
from string import whitespace

INPUT_HEIGHT = 32
INPUT_WIDTH = 32

def handler(data, context):
    """Handle request.
    Args:
        data (obj): the request data
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, (optional) response content type
    """
    processed_input = _process_input(data, context)
    print("sending processed_input to context.rest_uri:", context.rest_uri)
    response = requests.post(context.rest_uri, data=processed_input)
    return _process_output(response, context)


def _process_input(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API

    Args:
        data (obj): the request data stream
        context (Context): an object containing request and configuration details

    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """

    read_data = data.read()

    # endpoint API
    if context.request_content_type == 'application/json':
        # read as numpy array
        image_np = np.asarray(json.loads(read_data)).astype(np.dtype('uint8'))
        image_np = np.array(Image.fromarray(image_np).resize((INPUT_HEIGHT,INPUT_WIDTH)))
        
    # batch transform of jpegs
    elif context.request_content_type == 'application/x-image':
        # load image from bytes and resize
        image_from_bytes = Image.open(BytesIO(read_data)).convert('RGB')
        image_from_bytes = image_from_bytes.resize((INPUT_HEIGHT,INPUT_WIDTH))
        image_np = np.array(image_from_bytes)
    
    # batch transform of tfrecord
    elif context.request_content_type == 'application/x-tfexample':
        example = tf.train.Example()
        example.ParseFromString(read_data)
        example_feature = MessageToDict(example.features)
        image_encoded = str.encode(example_feature['feature']['image']['bytesList']['value'][0])
        image_b64 = base64.decodebytes(image_encoded)
        image_np = np.frombuffer(image_b64, dtype=np.dtype('uint8')).reshape(32,32,3)
        image_np = np.array(Image.fromarray(image_np).resize((INPUT_HEIGHT,INPUT_WIDTH)))
    
    # raise error if content type is not supported
    else:
        print("")
        _return_error(415, 'Unsupported content type "{}"'.format(
            context.request_content_type or 'Unknown'))
        

    # preprocess for resnet50
    image_np = tf.keras.applications.resnet_v2.preprocess_input(image_np)
    
    # json serialize
    data_np_json = {"instances": [image_np.tolist()]}
    data_np_json_serialized = json.dumps(data_np_json)

    return data_np_json_serialized


def _process_output(response, context):
    """Post-process TensorFlow Serving output before it is returned to the client.

    Args:
        response (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details

    Returns:
        (bytes, string): data to return to client, response content type
    """
    if response.status_code != 200:
        _return_error(response.status_code, response.content.decode('utf-8'))
        
    response_content_type = context.accept_header
    print("response.json():", response.json())
    
    # remove whitespace from output JSON string
    prediction = response.content.decode('utf-8').translate(dict.fromkeys(map(ord,whitespace)))
    
    return prediction, response_content_type


def _return_error(code, message):
    raise ValueError('Error: {}, {}'.format(str(code), message))