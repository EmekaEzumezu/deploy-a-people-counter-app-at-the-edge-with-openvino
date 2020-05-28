"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    global prob_threshold
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()
    input_shape = net_input_shape['image_tensor']

    ### TODO: Handle the input stream ###
    # Checks for live feed
    if args.input == 'CAM':
        input_stream = 0

    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_stream = args.input

    # Checks for video file
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)
    if input_stream:
        cap.open(args.input)

    if not cap.isOpened():
        log.error("ERROR! Unable to open video source")

    global initial_w, initial_h
    #prob_threshold = args.prob_threshold
    initial_w = cap.get(3)
    initial_h = cap.get(4)

    # Flag for the input image
    single_image_mode = False

    #iniatilize desired variables
    global total_count, report, duration, net_output, frame, prev_counter, prev_duration, counter, dur
    
    total_count = 0
    request_id=0
    
    report = 0


    prev_counter = 0
    prev_duration = 0
    counter = 0
    dur = 0


    # NOTE: Some code implementation gotten from 
    # https://github.com/prateeksawhney97/People-Counter-Application-Using-Intel-OpenVINO-Toolkit/blob/master/main.py

    ### TODO: Loop until stream is over ###
    while cap.isOpened():

        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (input_shape[3], input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        net_input = {'image_tensor': p_frame,'image_info': p_frame.shape[1:]}
        duration = None
        infer_network.exec_net(request_id, net_input)

        ### TODO: Wait for the result ###
        if infer_network.wait(request_id) == 0:

            ### TODO: Get the results of the inference request ###
            net_output = infer_network.get_output(request_id)

            ### TODO: Extract any desired stats from the results ###
            #probs = net_output[0, 0, :, 2]

            frame, net_output, prev_counter, prev_duration, counter, dur, report, total_count, duration = \
            desired_stats(frame, net_output, prev_counter, 
                prev_duration, counter, dur, report, total_count, duration)

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

            # When new person enters the video
            client.publish('person',
               json.dumps({
                   'count': report, 'total': total_count}),
               qos=0, retain=False)

            # Person duration in the video is calculated
            if duration is not None:
                client.publish('person/duration', 
                    json.dumps({'duration': duration}), 
                    qos=0, retain=False)


            if key_pressed == 27:
                break

        ### TODO: Send the frame to the FFMPEG server ###
        frame = cv2.resize(frame, (768, 432))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)

    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    infer_network.clean()

def desired_stats(frame, net_output, prev_counter, prev_duration, counter, dur, report, total_count, duration):

    # prev_counter = 0
    # prev_duration = 0
    pointer = 0
    # counter = 0
    # dur = 0
    # report = 0
    # total_count = 0
    # duration = None

    probs = net_output[0, 0, :, 2]

    for i, p in enumerate(probs):
        if p > prob_threshold:
            pointer += 1
            box = net_output[0, 0, i, 3:]
            p1 = (int(box[0] * initial_w), int(box[1] * initial_h))
            p2 = (int(box[2] * initial_w), int(box[3] * initial_h))
            frame = cv2.rectangle(frame, p1, p2, (0, 255, 0), 3)

    if pointer != counter:
        prev_counter = counter
        counter = pointer
        if dur >= 3:
            prev_duration = dur
            dur = 0
        else:
            dur = prev_duration + dur
            prev_duration = 0  # unknown, not needed in this case
    else:
        dur += 1
        if dur >= 3:
            report = counter
            if dur == 3 and counter > prev_counter:
                total_count += counter - prev_counter
            elif dur == 3 and counter < prev_counter:
                duration = int((prev_duration / 10.0) * 1000)

    return frame, net_output, prev_counter, prev_duration, counter, dur, report, total_count, duration


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
