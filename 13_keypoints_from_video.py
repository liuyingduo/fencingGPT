import sys
import cv2
import os
from sys import platform
import argparse
import time



try:
    # 导入 OpenPose
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        sys.path.append(r'D:\study\openpose-1.7.0\bin')
        os.environ['PATH'] = os.environ['PATH'] + r';D:\study\openpose-1.7.0\bin'
        import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found.')
        raise e

    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="../examples/media/fencing_01.mp4", help="Path to video file.")
    parser.add_argument("--write_json", default="./output_jsons/", help="Directory to write JSON output.")
    parser.add_argument("--output_video", default="./fencing_01_openpose.mp4", help="Path to save the output video.")
    args = parser.parse_known_args()

    # 设置 OpenPose 参数
    params = {
        "model_folder": "../models/",
        "number_people_max": 2,
        "write_json": args[0].write_json
    }

    # 初始化 OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # 打开视频文件
    cap = cv2.VideoCapture(args[0].video)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    # 获取视频信息以创建输出文件
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args[0].output_video, fourcc, fps, (frame_width, frame_height))

    start = time.time()

    # 处理视频的每一帧
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 创建 Datum 对象并处理当前帧
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        # 将处理后的帧写入输出视频
        out.write(datum.cvOutputData)

        # 在窗口中显示结果（可选）
        cv2.imshow("OpenPose", datum.cvOutputData)
        key = cv2.waitKey(15)
        if key == 27:  # 按 Esc 退出
            break

    cap.release()
    out.release()
    end = time.time()
    print("OpenPose video processing finished. Total time: " + str(end - start) + " seconds")
except Exception as e:
    print(e)
    sys.exit(-1)
