import cv2
import multiprocessing
import sys
import argparse
import setproctitle as setproctitle
import decoding.handler as dec_handler
import inference.handler as inf_handler
import visualizer.handler as vis_handler


if __name__=="__main__":
    setproctitle.setproctitle("Main")

    parser = argparse.ArgumentParser()

    parser.add_argument('--input',
                        type=str,
                        required=False,
                        default='rtsp://10.10.100.12:8555/proxied12',
                        help='Input file to load from in image or video mode')
    parser.add_argument('-m1',
                        '--model1',
                        type=str,
                        required=False,
                        default='model1',
                        help='Inference model name')
    parser.add_argument('-m2',
                        '--model2',
                        type=str,
                        required=False,
                        default='model2',
                        help='Inference model name, default vkd')
    parser.add_argument('--width',
                        type=int,
                        required=False,
                        default=640,
                        help='Inference model input width, default 608')
    parser.add_argument('--height',
                        type=int,
                        required=False,
                        default=640,
                        help='Inference model input height, default 608')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8221',
                        help='Inference server URL, default localhost:8001')

    parser.add_argument('-i',
                        '--model-info',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Print model status, configuration and statistics')
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose client output')
    parser.add_argument('-t',
                        '--client-timeout',
                        type=float,
                        required=False,
                        default=None,
                        help='Client timeout in seconds, default no timeout')
    parser.add_argument('-s',
                        '--ssl',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable SSL encrypted channel to the server')
    parser.add_argument('-r',
                        '--root-certificates',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded root certificates, default none')
    parser.add_argument('-p',
                        '--private-key',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded private key, default is none')
    parser.add_argument('-x',
                        '--certificate-chain',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded certicate chain default is none')

    FLAGS = parser.parse_args()
    print(FLAGS)
    INPUT_STREAM = FLAGS.input

    multiprocessing.set_start_method('spawn')

    framesQueue = multiprocessing.Queue()

    predictionsQueue = multiprocessing.Queue()

    decodingProcess = multiprocessing.Process(target=dec_handler.run, args=(framesQueue, INPUT_STREAM, FLAGS))
    #
    inferenceProcess = multiprocessing.Process(target=inf_handler.run, args=(framesQueue, predictionsQueue, FLAGS))

    visualizerProcess = multiprocessing.Process(target=vis_handler.run, args=(predictionsQueue, FLAGS))

    decodingProcess.start()
    inferenceProcess.start()
    visualizerProcess.start()


    decodingProcess.join()
    inferenceProcess.join()



    #visualizerProcess.join()


