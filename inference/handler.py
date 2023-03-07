import cv2
import setproctitle as setproctitle
import onnxruntime
import time
import tritonclient.grpc as grpcclient
from utils.utils import *
import sys


fp16 = False
img_size = 640
nhwc = True
frameIndex = 0
stride1 = 32
auto = False
w = 1080
h = 1920

wh = w * h
hwhh = int(wh / 4)
dh = h + int(h / 2)
SHAPE = [1088, 1920]
ACQUISITIONSIZE_Y = (3, SHAPE[0] * SHAPE[1])
ACQUISITIONSIZE_U = (3, ACQUISITIONSIZE_Y[1] // 4)
ACQUISITIONSIZE_V = (3, ACQUISITIONSIZE_U[1])


def run(framesQueue, predictionQueue, FLAGS):
    setproctitle.setproctitle("Inference")
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=FLAGS.url,
            verbose=FLAGS.verbose,
            ssl=FLAGS.ssl,
            root_certificates=FLAGS.root_certificates,
            private_key=FLAGS.private_key,
            certificate_chain=FLAGS.certificate_chain)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

        # Health check
    if not triton_client.is_server_live():
        print("FAILED : is_server_live")
        sys.exit(1)

    if not triton_client.is_server_ready():
        print("FAILED : is_server_ready")
        sys.exit(1)

    if not triton_client.is_model_ready(FLAGS.model1):
        print("FAILED : is_model_ready", FLAGS.model1)
        sys.exit(1)
    if not triton_client.is_model_ready(FLAGS.model2):
        print("FAILED : is_model_ready", FLAGS.model2)
        sys.exit(1)


    while True:

        try:

            inputs1 = []

            outputs1 = []

            inputs1.append(grpcclient.InferInput('images', [1, 3, 640, 640], "FP32"))
            outputs1.append(grpcclient.InferRequestedOutput('output0'))

            inputs2 = []

            outputs2 = []

            inputs2.append(grpcclient.InferInput('images', [1, 3, 640, 640], "FP32"))
            outputs2.append(grpcclient.InferRequestedOutput('output0'))


            element = framesQueue.get()
            im = element.copy()
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            t_inf = time.time()
            im = letterbox(im, img_size, stride=stride1, auto=auto)[0]  # padded resize

            # im = im.reshape([1, *im.shape])  # HW to CHW

            im = np.ascontiguousarray(im).astype(np.float32)
            im = im.transpose(2, 0, 1)
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            inputs1[0].set_data_from_numpy(im)

            results1 = triton_client.infer(model_name=FLAGS.model1,
                                          inputs=inputs1,
                                          outputs=outputs1
                                          )

            y1 = results1.as_numpy('output0')



            results2 = triton_client.infer(model_name=FLAGS.model2,
                                          inputs=inputs1,
                                          outputs=outputs2
                                          )

            y2 = results2.as_numpy('output0')

            t3 = time.time()

            outputs1 = nms(y1, nclasses=80)
            outputs2 = nms(y2, nclasses=80)



            predictionQueue.put({"m": "prd", "c1": outputs1, "c2": outputs2, "frame": element},
                          block=False)

            print("Time inference both models", t3-t_inf)

        except:
            pass
