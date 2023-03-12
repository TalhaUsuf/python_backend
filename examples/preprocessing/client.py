
import os, sys
import numpy as np
import json

import tritonclient.http as httpclient
import torch
import argparse
from PIL import Image

def load_image(img_path: str):
    """
    Loads an encoded image as an array of bytes.
    
    """
    return np.fromfile(img_path, dtype='uint8')

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords
def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2



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
    # parser.add_argument("--url",
    #                     type=str,
    #                     required=False,
    #                     default="localhost:8001",
    #                     help="Inference server URL. Default is localhost:8001.")
    parser.add_argument('-v',
                        "--verbose",
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-t', '--type', type=str, default='http', help='http or grpc')
    parser.add_argument('--url_http', type=str, default="localhost:8000", help='url to triton http server')
    # parser.add_argument('--url_grpc', type=str, default="localhost:8001", help='url to triton grpc server')
    parser.add_argument(
        "--label_file",
        type=str,
        default="./model_repository/resnet50_trt/labels.txt",
        help="Path to the file with text representation of available labels")
    args = parser.parse_args()

    # try:
    #     triton_client = httpclient.InferenceServerClient(
    #             url=args.url, verbose=args.verbose)
    # except Exception as e:
    #     print("channel creation failed: " + str(e))
    #     sys.exit(1)

    # with open(args.label_file) as f:
    #     labels_dict = {idx: line.strip() for idx, line in enumerate(f)}

    
    # if args.type=="grpc":
    #     triton_client = tritongrpcclient.InferenceServerClient(
    #         url=args.url_grpc, verbose=args.verbose)
    
    
    if args.type=="http":
        triton_client = httpclient.InferenceServerClient(
            url=args.url_http, verbose=args.verbose)
        
    
    
    # ==========================================================================
    #                             preprocessing                                   
    # ==========================================================================
    if args.model_name == "yolo_preprocess":
        
        # triton_client = httpclient.InferenceServerClient(
        #     url=args.url, verbose=args.verbose)
        
        inputs = []
        outputs = []
    
        input_name = "pre_input"
        output_name = "pre_output"
        # read image as bytes
        image_data = load_image(args.image) # shape (1005970,)
        # convert to batch
        image_data = np.expand_dims(image_data, axis=0)  # shape (1, 1005970)
        # create input and output tensors
        # inputs.append(tritongrpcclient.InferInput(input_name, image_data.shape, "UINT8"))
        # outputs.append(tritongrpcclient.InferRequestedOutput(output_name))
        inputs.append(httpclient.InferInput(input_name, image_data.shape, "UINT8"))
        outputs.append(httpclient.InferRequestedOutput(output_name))
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
        # triton_client = tritongrpcclient.InferenceServerClient(
        #     url=args.url, verbose=args.verbose)
        
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
            httpclient.InferInput(input_name, image_data.shape, "FP32"))
        outputs.append(httpclient.InferRequestedOutput(output_name))
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
        
    if args.model_name == "yolo_postprocess":
    # --------------------------------------------------------------------------
    #   🔴                 POST PROCESSING after this                        
    # --------------------------------------------------------------------------
        inputs = []
        outputs = []
        input_name = "post_input"
        
        output_name = "post_output"
        
        # create dummy input as yolo output shape
        sample_input = np.random.rand(1, 25200, 85).astype(np.float32)
        
        

        inputs.append(
            httpclient.InferInput(input_name, sample_input.shape, "FP32"))
        outputs.append(httpclient.InferRequestedOutput(output_name))
        

        
        inputs[0].set_data_from_numpy(sample_input)
        
        results = triton_client.infer(model_name=args.model_name,
                                    inputs=inputs,
                                    outputs=outputs)

        # output0_data = results.as_numpy(output_name)
        output0_data = results.as_numpy(output_name)
        print(output0_data.shape)
        
    if args.model_name == "yolo_ensemble":
    # --------------------------------------------------------------------------
    #   🔴                 ENSEMBLE                        
    # --------------------------------------------------------------------------
        inputs = []
        outputs = []
        input_name = "INPUT_ENSEMBLE"
        
        output_name = "OUTPUT_ENSEMBLE"
        
        sz = Image.open(args.image).size
        image_data = load_image(args.image) # shape (1005970,)
        # convert to batch
        image_data = np.expand_dims(image_data, axis=0)  # shape (1, 1005970)

        inputs.append(httpclient.InferInput(input_name, image_data.shape, "UINT8"))
        outputs.append(httpclient.InferRequestedOutput(output_name))

        

        
        inputs[0].set_data_from_numpy(image_data)
        
        results = triton_client.infer(model_name=args.model_name,
                                    inputs=inputs,
                                    outputs=outputs)

        
        output0_data = results.as_numpy(output_name)
        # import joblib
        
        # joblib.dump(output0_data, "ensemble_out.pkl")
        # print(output0_data.shape)
        
        
        # # --------------------------------------------------------------------------
        # #                        map detections to labels                        
        # # --------------------------------------------------------------------------
        
        for i, det in enumerate(output0_data):
            # im0 --> resized image
            # im --> original image
            det[:, :4] = scale_coords([640, 640], det[:, :4], sz).round()
        # print(det)
        
        
        # #annotate the image
        for *xyxy, conf, _cls in reversed(det):
            c = int(_cls)  # integer class
            # label = f'{names[c]} {conf:.2f}'
            print(f"detected : {c} with conf. {conf}, coordinates : {xyxy}")
            # annotator.box_label(xyxy, label, color=colors(c, True))
        
        
    
    # maxs = np.argmax(output0_data, axis=1)
    # print(maxs)
    # print("Result is class: {}".format(labels_dict[maxs[0]]))
