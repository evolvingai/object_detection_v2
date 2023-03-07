

This small project can be used to detect 80 classes (coco classes) in a stream or in a video file.
For Decoding I used OPENCV, because there were not requirements for optimiztion,
however in a deployable system I would choose DALI from NVIDIA or PyAv with custom decoding.

I used a process for each major task
    one process for decoding, grabing each frame, decode it and resize
    one process for inference (using 2 different models - a light one and a heavier one)
    one process for visualization (opencv drawing functions can be heavy)

  Decoding process puts the frames in a multiprocessing queue.
  Inference process gets frames from the multiprocessing queue, preprocess the frame, then each model does predictions on that frame sequentially, using TRITON server.
  The process comunicates through grpc with the triton server.
  Models are in the onnx format (fp32). In the zip provided there are two option (onnx for gpu or onnx for cpu)
  At the end the predictions from both models are put in a multiprocessing queue.
  Visualizer process gets data (frame and predictions from the predictions queue). For each frame we have the bounding boxes, the classes, the confidences.
  In two different windows the output of the models are displayed.


HOW TO RUN:

pip install -r requirements.txt

FLAGS:
--input : rtsp or file (mp4, avi, etc)
--model1 : model name in triton server (in my case it was names model1 and this is the default value)
--model2 : model name in triton server (in my case it was names model2 and this is the default value)
--url : triton server url and port (default 'localhost:8221') (as the triton server was on the same machine on which I was running the code)
others can be default

In order to setup triton server you need nvidia-docker2

you need in the triton_deploy/models/ the following dirs
    model1/1/model.onnx (onnx of the first model found in the cpu_onnx as yolov5s.onnx)
    model2/1/model.onnx (onnx of the second model found in the cpu_onnx as yolov5x.onnx)

for starting a triton server (assuimg you have installed nvidia-docker2)
you need to run:
sudo docker run -d  --name tritonserver-23.01 --shm-size=1g --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -p8220:8000 -p8221:8001 -p8222:8002 -v$(pwd)/triton_deploy/models:/models nvcr.io/nvidia/tritonserver:23.01-py3 tritonserver --model-repository=/models --strict-model-config=false --grpc-infer-allocation-pool-size=16
from a folder containing triton_deploy/models

python main.py --input rtsp://86.44.41.160:554/axis-media/media.amp

