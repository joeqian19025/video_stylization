import cv2
from streamer import StreamerServer
from ReReVST.test.framework import Stylization
import torch
import json
from utils import enpack, depack, ReshapeTool

address = 'localhost'
port = 6666

style_img = 'ReReVST/test/inputs/starry_night.jpg'
model_path = 'ReReVST/test/Model/style_net-TIP-final.pth'
cuda = torch.cuda.is_available()
use_global = True
interval = 100



class RealTimeStylization:
    framework = None
    style = None
    interval = 0

    total_frames = 0

    reshape = ReshapeTool()

    def __init__(self, style_img, model_path, cuda, use_global, interval):
        self.framework = Stylization(model_path, cuda, use_global)
        self.style = cv2.imread(style_img)
        self.framework.prepare_style(self.style)
        self.framework.clean()
        self.interval = interval

    def render(self, frame):
        H, W, C = frame.shape
        if use_global:
            if self.total_frames % self.interval == 0:
                self.framework.clean()
                self.framework.add(frame)
                self.framework.compute()
        
        reshaped_frame = self.reshape.process(frame)
        styled_frame = self.framework.transfer(reshaped_frame)
        self.total_frames = self.total_frames + 1
        return styled_frame[64:64+H,64:64+W,:]


stylizer = RealTimeStylization(style_img, model_path, cuda, use_global, interval)

print('stylizer: ', stylizer)

streamer = StreamerServer((address, port), discard_older=3)

while True:
    # Recieiving a frame from the client
    input_msg = streamer.recv()
    if input_msg != None:
        print('Recieved a frame')
        input_frame = depack(input_msg)

        # Process the frame
        # res = cv2.flip(input_frame, 0)
        rendered_frame = stylizer.render(input_frame)

        # Send back the processed frame to the client
        msg = enpack(rendered_frame)
        try:
            streamer.send(msg)
        except RuntimeError as e:
            print('CLIENT <send> ERROR: ', e)