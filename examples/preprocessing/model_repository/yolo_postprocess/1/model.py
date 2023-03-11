
import numpy as np
import sys
import json
import io
from rich.console import Console
import cv2
from PIL import Image
# Reference for pb_utils:
# 🔖 https://github.com/triton-inference-server/python_backend/blob/main/src/resources/triton_python_backend_utils.py
import triton_python_backend_utils as pb_utils
import torch
import torchvision

class TritonPythonModel:
    """
    Postprocessing applies NMS to each image. Since the model will run in ensemble mode and post-process will have
    no connection with pre-process and hence not getting the original image size. So bbox--> real coordinate scaling will be done
    on the request api side.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "post_output")

        # Triton dtypes (from pbtxt file) ➡️ numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

    def execute(self, requests):
        """
        Applies preprocessing steps to the input requests (received as byte arrays)
        and converts them to the format expected by the model after applying preprocessing steps.


        Notes:
        ------
        For reference of pb_utils , see : 🔗 `https://github.com/triton-inference-server/python_backend/blob/main/src/resources/triton_python_backend_utils.py`
        

        Parameters
        ----------
        requests : list
            A list of pb_utils.InferenceRequest

        Returns
        -------
        list
            A list of pb_utils.InferenceResponse. The length of this list must
            be the same as `requests`
        """
        
        output0_dtype = self.output0_dtype
        # responses should hold a list of pb_utils.InferenceResponse objects 
        responses = []
        
        for request in requests:
            # 💀 Each request will correspond to a single input image
            # Get pre_input tensor from the request read as bytes
            in_0 = pb_utils.get_input_tensor_by_name(request, "post_input")
            in_0 = in_0.as_numpy()
            # np.ndarray ➡️ torch tensor since the NMS is taken from torch
            in_0 = torch.from_numpy(in_0).to(torch.device('cpu'))
            pred = self.non_max_suppression(in_0, conf_thres = 0.25, iou_thres = 0.45, agnostic= False, max_det=1000)
            # convert to the type expected by this model
            nmsed_detections = np.array([np.array(k.tolist()).squeeze() for k in pred])



            
            #🔴 output0_dtype is the output dtype (np format) as decoded from config.pbtxt
            out_tensor_0 = pb_utils.Tensor("post_output",
                                           nmsed_detections.astype(output0_dtype))

            
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            
            
            responses.append(inference_response)

        
        return responses


    
    def non_max_suppression(self,
                            prediction,
                            conf_thres=0.25,
                            iou_thres=0.45,
                            agnostic=False,
                            max_det=300):
        
        # ==========================================================================
        #                        define sone helper functions                                  
        # ==========================================================================
        def xywh2xyxy(x):
            # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
            y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
            y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
            y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
            y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
            y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
            return y
        def box_area(box):
            # box = xyxy(4,n)
            return (box[2] - box[0]) * (box[3] - box[1])
        def box_iou(box1, box2, eps=1e-7):
            # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
            (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
            inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

            # IoU = inter / (area1 + area2 - inter)
            return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter + eps)
        
        
        # # --------------------------------------------------------------------------
        # #                              actual NMS starts here                        
        # # --------------------------------------------------------------------------
        
        bs = prediction.shape[0]  # batch size
        xc = prediction[..., 4] > conf_thres  # candidates
        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        redundant = True  # require redundant detections
        merge = False  # use merge-NMS
        output = [torch.zeros((0, 6), device = prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence
            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
        return output





    def finalize(self):
        """upon model unloading, since for the task model is being loaded in polling mode so this function will
        cannot be called from request
        """
        print('👉\tUnloading the yolo_postprocessing function')