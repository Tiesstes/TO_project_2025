from ultralytics import YOLO
from PIL import Image
import os

#model = YOLO("yolo11s.pt")
model = YOLO("trained_yolo11s.pt")


image_results = []
video_results = []

for i in range(257, 271, 1):
    image = Image.open(f"D:/Informatyka/PyCharmProjects/Traffic_Signs_NN/datasets_augmented/misc/images/IMG_0{i}.JPG")
    rotated_image = image.rotate(-90)
    model(rotated_image, imgsz=[736, 1280], save=True, name=f"IMG_0{i}_result")


model.track(imgsz = [736, 1280], source = "D:/Informatyka/PyCharmProjects/Traffic_Signs_NN/datasets_augmented/misc/video/MUTE_287_resized.MOV", show=True)
model.track(imgsz = [736, 1280], source = "D:/Informatyka/PyCharmProjects/Traffic_Signs_NN/datasets_augmented/misc/video/MUTE_290_resized.MOV", show=True)
model.track(imgsz = [736, 1280], source = "D:/Informatyka/PyCharmProjects/Traffic_Signs_NN/datasets_augmented/misc/video/MUTE_291_resized.MOV", show=True)




