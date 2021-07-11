import torch
from torchsummary import summary

from nets.yolo4 import YoloBody

if __name__ == "__main__":
    # 需要使用device来指定网络在GPU还是CPU运行
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = YoloBody(3,2,backbone="mobilenetv1").to(device)
    summary(model, input_size=(3, 416, 416))
    
    # mobilenetv1-yolov4 40,952,893
    # mobilenetv2-yolov4 39,062,013
    # mobilenetv3-yolov4 39,989,933

    # 修改了panet的mobilenetv1-yolov4 12,271,999
    # 修改了panet的mobilenetv2-yolov4 10,381,119
    # 修改了panet的mobilenetv3-yolov4 11,309,039(43.14MB)
    # 修改了panet的mobilenetv3_0.75-yolov4 10,166,767(38.78MB)
    # 修改了panet的mobilenetv3_0.5-yolov4 9,188,743(35.05MB)