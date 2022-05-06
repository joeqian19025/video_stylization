import cv2
from streamer import StreamerClient
import time
import json

from utils import depack, enpack

address = 'localhost'
port = 6666

streamer = StreamerClient((address, port), discard_older = 3)

cap = cv2.VideoCapture(0)

last_frame = None

if cap.isOpened():
    _, last_frame = cap.read()

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (300, 300))
    send_msg = enpack(frame)    
    try:
        streamer.send(send_msg)
    except RuntimeError as e:
        print('SERVER <send> ERROR: ', e)

    ret_msg = streamer.recv()
    if ret_msg != None:
        last_frame = depack(ret_msg)
    
    cv2.imshow('Real Time Stylized Camera', last_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break