
import os, sys
import numpy as np
import json

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
import tritongrpcclient

import argparse
from PIL import Image

def load_image(img_path: str):
    """
    Loads an encoded image as an array of bytes.
    
    """
    return np.fromfile(img_path, dtype='uint8')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str,
                        required=False,
                        default="ensemble_python_resnet50",
                        help="Model name")
    parser.add_argument("--image",
                        type=str,
                        required=True,
                        help="Path to the image")
    parser.add_argument("--url",
                        type=str,
                        required=False,
                        default="localhost:8001",
                        help="Inference server URL. Default is localhost:8001.")
    parser.add_argument('-v',
                        "--verbose",
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument(
        "--label_file",
        type=str,
        default="./model_repository/resnet50_trt/labels.txt",
        help="Path to the file with text representation of available labels")
    args = parser.parse_args()

    try:
        triton_client = httpclient.InferenceServerClient(
                url=args.url, verbose=args.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    with open(args.label_file) as f:
        labels_dict = {idx: line.strip() for idx, line in enumerate(f)}

    
    
    # ==========================================================================
    #                             preprocessing                                   
    # ==========================================================================
    if args.model_name == "yolo_preprocess":
        
        triton_client = tritongrpcclient.InferenceServerClient(
            url=args.url, verbose=args.verbose)
        
        inputs = []
        outputs = []
    
        input_name = "pre_input"
        output_name = "pre_output"
        # read image as bytes
        image_data = load_image(args.image) # shape (1005970,)
        # convert to batch
        image_data = np.expand_dims(image_data, axis=0)  # shape (1, 1005970)
        # create input and output tensors
        inputs.append(tritongrpcclient.InferInput(input_name, image_data.shape, "UINT8"))
        outputs.append(tritongrpcclient.InferRequestedOutput(output_name))
        # inputs.append(httpclient.InferInput(input_name, image_data.shape, "UINT8"))
        # outputs.append(httpclient.InferRequestedOutput(output_name))
        # put data into input tensor
        inputs[0].set_data_from_numpy(image_data)
        # run inference
        results = triton_client.infer(model_name=args.model_name,
                                    inputs=inputs,
                                    outputs=outputs,
                                    # model_version="1"
                                    )

        
        
        
        # get output tensor
        results = results.as_numpy(output_name)
        print(results)
        
    if args.model_name == "yolov5":
        # ==========================================================================
        #                                yolov5                                   
        # ==========================================================================
        triton_client = tritongrpcclient.InferenceServerClient(
            url=args.url, verbose=args.verbose)
        
        inputs = []
        outputs = []
        input_name = "images"
        
        output_name = "output0"
        # breakpoint()
        image_data = np.array(Image.open(args.image).resize((640,640)).convert('RGB'))
        # convert HWC to CHW format
        image_data = (image_data.transpose((2, 0, 1)) / 255.0).astype(np.float32)
        # image_data = load_image(args.image) # shape (1005970,)
        image_data = np.expand_dims(image_data, axis=0)  # shape (1, 1005970)

        inputs.append(
            tritongrpcclient.InferInput(input_name, image_data.shape, "FP32"))
        outputs.append(tritongrpcclient.InferRequestedOutput(output_name))
        # inputs.append(
        #     tritongrpcclient.InferInput(input_name, image_data.shape, "UINT8"))
        # outputs.append(tritongrpcclient.InferRequestedOutput(output_name))

        # inputs[0].set_data_from_numpy(image_data)
        inputs[0].set_data_from_numpy(image_data)
        # results = triton_client.infer(model_name=args.model_name,
        #                               inputs=inputs,
        #                               outputs=outputs)
        results = triton_client.infer(model_name=args.model_name,
                                    inputs=inputs,
                                    outputs=outputs)

        # output0_data = results.as_numpy(output_name)
        output0_data = results.as_numpy(output_name)
        print(output0_data.shape)
        
        
    # # --------------------------------------------------------------------------
    # #   🔴                 POST PROCESSING after this                        
    # # --------------------------------------------------------------------------
    
    
    # maxs = np.argmax(output0_data, axis=1)
    # print(maxs)
    # print("Result is class: {}".format(labels_dict[maxs[0]]))
