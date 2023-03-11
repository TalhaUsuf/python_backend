# Yolov5 models

## Torch to onnx

```bash
cd Yolov5-ONNX/exporter/yolov5
```

Convert torch ckpt to onnx model:

```bash
python python  export.py  --weights yolov5n.pt  --dynamic  --include onnx 
python python  export.py  --weights yolov5s.pt  --dynamic  --include onnx 
python python  export.py  --weights yolov5m.pt  --dynamic  --include onnx 
```


After this step, following files will be present:
 [x] yolov5n.onnx
 [x] yolov5s.onnx
 [x] yolov5m.onnx



## Onnx detection run check

```bash
cd ../../
python detect.py --input ../mug.jpg --output detection.jpg  --weights "exporter/yolov5/yolov5m.onnx"
```

## Checking the inputs using Polygraphy


```bash
polygraphy   inspect model   exporter/yolov5/yolov5n.onnx
```

```markdown
[I] Loading model: /home/fsuser/python_backend/examples/preprocessing/Yolov5-ONNX/exporter/yolov5/yolov5n.onnx
[I] ==== ONNX Model ====
    Name: torch_jit | ONNX Opset: 17
    
    ---- 1 Graph Input(s) ----
    {images [dtype=float32, shape=('batch', 3, 'height', 'width')]}
    
    ---- 1 Graph Output(s) ----
    {output0 [dtype=float32, shape=('batch', 'anchors', 'Concatoutput0_dim_2')]}
    
    ---- 123 Initializer(s) ----
    
    ---- 524 Node(s) ----
```


```bash
polygraphy   inspect model   exporter/yolov5/yolov5s.onnx
```

```markdown
[I] Loading model: /home/fsuser/python_backend/examples/preprocessing/Yolov5-ONNX/exporter/yolov5/yolov5s.onnx
[I] ==== ONNX Model ====
    Name: torch_jit | ONNX Opset: 17
    
    ---- 1 Graph Input(s) ----
    {images [dtype=float32, shape=('batch', 3, 'height', 'width')]}
    
    ---- 1 Graph Output(s) ----
    {output0 [dtype=float32, shape=('batch', 'anchors', 'Concatoutput0_dim_2')]}
    
    ---- 123 Initializer(s) ----
    
    ---- 524 Node(s) ----
```


```bash
polygraphy   inspect model   exporter/yolov5/yolov5m.onnx
```

```markdown
[I] Loading model: /home/fsuser/python_backend/examples/preprocessing/Yolov5-ONNX/exporter/yolov5/yolov5m.onnx
[I] ==== ONNX Model ====
    Name: torch_jit | ONNX Opset: 17
    
    ---- 1 Graph Input(s) ----
    {images [dtype=float32, shape=('batch', 3, 'height', 'width')]}
    
    ---- 1 Graph Output(s) ----
    {output0 [dtype=float32, shape=('batch', 'anchors', 'Concatoutput0_dim_2')]}
    
    ---- 167 Initializer(s) ----
    
    ---- 597 Node(s) ----
```


So in all cases, following info is needed for the `config.pbtxt`:
 - {images [dtype=float32, shape=('batch', 3, 'height', 'width')]}
 - {output0 [dtype=float32, shape=('batch', 'anchors', 'Concatoutput0_dim_2')]}







# **Preprocessing Using Python Backend Example**
This example shows how to preprocess your inputs using Python backend before it is passed to the TensorRT model for inference. This ensemble model includes an image preprocessing model (preprocess) and a TensorRT model (resnet50_trt) to do inference.

**1. Converting PyTorch Model to ONNX format:**

Run onnx_exporter.py to convert ResNet50 PyTorch model to ONNX format. Width and height dims are fixed at 224 but dynamic axes arguments for dynamic batching are used. Commands from the 2. and 3. subsections shall be executed within this Docker container.

    $ docker run -it --gpus=all -v $(pwd):/workspace nvcr.io/nvidia/pytorch:xx.yy-py3 bash
    $ pip install numpy pillow torchvision
    $ python onnx_exporter.py --save model.onnx
    
**2. Create the model repository:**

    $ mkdir -p model_repository/ensemble_python_resnet50/1
    $ mkdir -p model_repository/preprocess/1
    $ mkdir -p model_repository/resnet50_trt/1
    
    # Copy the Python model
    $ cp model.py model_repository/preprocess/1

**3. Build a TensorRT engine for the ONNX model**

Set the arguments for enabling fp16 precision --fp16. To enable dynamic shapes use --minShapes, --optShapes, and maxShapes with --explicitBatch:

    $ trtexec --onnx=model.onnx --saveEngine=./model_repository/resnet50_trt/1/model.plan --explicitBatch --minShapes=input:1x3x224x224 --optShapes=input:1x3x224x224 --maxShapes=input:256x3x224x224 --fp16

**4. Run the command below to start the server container:**

Under python_backend/examples/preprocessing, run this command to start the server docker container:

    $ docker run --gpus=all -it --rm -p8000:8000 -p8001:8001 -p8002:8002 -v$(pwd):/workspace/ -v/$(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:xx.yy-py3 bash
    $ pip install numpy pillow torchvision
    $ tritonserver --model-repository=/models
     
**5. Start the client to test:**

Under python_backend/examples/preprocessing, run the commands below to start the client Docker container:

    $ wget https://raw.githubusercontent.com/triton-inference-server/server/main/qa/images/mug.jpg -O "mug.jpg"
    $ docker run --rm --net=host -v $(pwd):/workspace/ nvcr.io/nvidia/tritonserver:xx.yy-py3-sdk python client.py --image mug.jpg 
    $ The result of classification is:COFFEE MUG    

Here, since we input an image of "mug" and the inference result is "COFFEE MUG" which is correct.
