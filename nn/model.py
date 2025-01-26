import multiprocessing

from torch.nn.functional import dropout
from ultralytics import YOLO
import torch, torchvision
from multiprocessing import Pool




EPOCHS = 75



if __name__ == '__main__':

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(torch.cuda.get_device_name(0))
    print(f"CUDA version: {torch.version.cuda}")
    print(f"torchvision version: {torchvision.__version__}")


    multiprocessing.freeze_support()
    main_dir = "D:\\Informatyka\\PyCharmProjects\\Traffic_Signs_NN"

    nn_model = YOLO("yolo11s.pt")

    results = nn_model.train(data= main_dir + "\\datasets_augmented\\data.yaml",
                   epochs=EPOCHS, patience=0, iou=0.65, pretrained=True, dropout=0.2, batch=0.85, imgsz=1024, device=torch.device("cuda"))

    metrics = nn_model.val()
    nn_model.save(main_dir + "\\nn\\trained_yolo11s.pt")

