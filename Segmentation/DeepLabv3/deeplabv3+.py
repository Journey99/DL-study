'''
Segmentation Models Pytorch (smp) 

! pip install segmentation-models-pytorch

'''
import segmentation_models_pytorch as smp

model = smp.DeepLabV3Plus(
    encoder_name="resnet101",        # backbone
    encoder_weights="imagenet",      # pretrained weights
    in_channels=3,
    classes=21,                      # 클래스 수
)


'''
MMSegmentation (OpenMMLab)

! pip install mmcv-full
! pip install mmsegmentation

'''

from mmseg.apis import init_segmentor, inference_segmentor

config = 'configs/deeplabv3plus/deeplabv3plus_r101-d8_512x512_80k_ade20k.py'
checkpoint = 'checkpoints/deeplabv3plus_r101.pth'

model = init_segmentor(config, checkpoint, device='cuda:0')
result = inference_segmentor(model, 'test.jpg')



'''
빠른 프로토타입 : torchvision
custom trainig : smp
production/reserach : mmsegmentation
최신 transformer 모델 : huggingface
'''