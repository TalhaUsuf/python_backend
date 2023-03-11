
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
import joblib


from torchvision import transforms


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
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
            model_config, "pre_output")

        # Triton dtypes ➡️ numpy types
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
            in_0 = pb_utils.get_input_tensor_by_name(request, "pre_input")
            # in_0.as_numpy() is (1, 1005970), np.array - uint8

            # 🔴 convert bytes to numpy array
            # joblib.dump(in_0.as_numpy(), "in_0.joblib")  
            img = in_0.as_numpy() # (1, 1005970), np.array - uint8
            # bytes to PIL image with RGB format
            image = Image.open(io.BytesIO(img.tobytes()))
            image = np.array(image) # image --> [H, W, C], (2521, 3361, 3)
            #Console().log(f"🔵\t[green]converted input request data into np.array")
            
            #Console().log(f"🔵\t[green]original image dims. [red]{_H}x{_W}")
            # resize image while maintaining aspect ratio, by adding border
            processed_image = self.letterbox(image, (640,640), stride=32, auto=False)
            #Console().log(f"🔵\t[green]image has been letter-boxed to [red]640x640 [green] dims.")
            # HWC ➡️ CHW and BGR ➡️ RGB
            processed_image = processed_image.transpose((2, 0, 1))[::-1]
            processed_image = np.ascontiguousarray(processed_image)
            # normalize pixel values to 0 - 1 range
            processed_image = processed_image/255.0
            if len(processed_image.shape) == 3:
                # convert to a rank-4 array
                processed_image = np.expand_dims(processed_image, 0)
            #Console().log(f"🔵\t[green]converted pixle values to [red]0-1 [green]range")
            # output dtype needs to be float32 as specified in config.pbtxt file
            processed_image = processed_image.astype(np.float32)
            
            #🔴 output0_dtype is the output dtype (np format) as decoded from config.pbtxt
            out_tensor_0 = pb_utils.Tensor("pre_output",
                                           processed_image.astype(output0_dtype))

            
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            
            
            responses.append(inference_response)

        
        return responses


    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        '''
        applies the letterbox operation to the input image, aspect ration is preserved.

        Parameters
        ----------
        im : np.ndarray
            _description_
        new_shape : tuple, optional
            _description_, by default (640, 640)
        color : tuple, optional
            _description_, by default (114, 114, 114)
        auto : bool, optional
            _description_, by default True
        scaleFill : bool, optional
            _description_, by default False
        scaleup : bool, optional
            _description_, by default True
        stride : int, optional
            _description_, by default 32

        Returns
        -------
        np.ndarray
            scaled image
        '''        
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im







    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('👉\tUnloading the yolo_preprocess function')
        
        
    # def execute(self, requests):
    #     """`execute` MUST be implemented in every Python model. `execute`
    #     function receives a list of pb_utils.InferenceRequest as the only
    #     argument. This function is called when an inference request is made
    #     for this model. Depending on the batching configuration (e.g. Dynamic
    #     Batching) used, `requests` may contain multiple requests. Every
    #     Python model, must create one pb_utils.InferenceResponse for every
    #     pb_utils.InferenceRequest in `requests`. If there is an error, you can
    #     set the error argument when creating a pb_utils.InferenceResponse

    #     Parameters
    #     ----------
    #     requests : list
    #       A list of pb_utils.InferenceRequest

    #     Returns
    #     -------
    #     list
    #       A list of pb_utils.InferenceResponse. The length of this list must
    #       be the same as `requests`
    #     """

    #     output0_dtype = self.output0_dtype

    #     responses = []

    #     # Every Python backend must iterate over everyone of the requests
    #     # and create a pb_utils.InferenceResponse for each of them.
    #     for request in requests:
    #         # Get INPUT0
    #         in_0 = pb_utils.get_input_tensor_by_name(request, "pre_input")

    #         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                          std=[0.229, 0.224, 0.225])

    #         loader = transforms.Compose([
    #             transforms.Resize([224, 224]),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(), normalize
    #         ])

    #         def image_loader(image_name):
    #             image = loader(image_name)
    #             #expand the dimension to nchw
    #             image = image.unsqueeze(0)
    #             return image

    #         # 🔴 convert bytes to numpy array
    #         img = in_0.as_numpy()
    #         image = Image.open(io.BytesIO(img.tobytes()))
    #         print(f"-------------- {np.array(image).shape}-----------------------")
    #         img_out = image_loader(image)
    #         img_out = np.array(img_out)

    #         out_tensor_0 = pb_utils.Tensor("pre_output",
    #                                        img_out.astype(output0_dtype))

    #         # Create InferenceResponse. You can set an error here in case
    #         # there was a problem with handling this inference request.
    #         # Below is an example of how you can set errors in inference
    #         # response:
    #         #
    #         # pb_utils.InferenceResponse(
    #         #    output_tensors=..., TritonError("An error occured"))
    #         inference_response = pb_utils.InferenceResponse(
    #             output_tensors=[out_tensor_0])
    #         responses.append(inference_response)

    #     # You should return a list of pb_utils.InferenceResponse. Length
    #     # of this list must match the length of `requests` list.
    #     return responses

    # def finalize(self):
    #     """`finalize` is called only once when the model is being unloaded.
    #     Implementing `finalize` function is OPTIONAL. This function allows
    #     the model to perform any necessary clean ups before exit.
    #     """
    #     print('Cleaning up...')
