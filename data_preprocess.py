from torchvision.transforms import v2
import cv2
import torch
import numpy as np


class Preprocessor:
    def augment_transform1(self):
        """Transform which does both random cropping and rotation along with preprocessing"""
        transform = v2.Compose([
        v2.Resize((512, 512)), 
        v2.Lambda(lambda img: img.convert("RGB")),  
        v2.Lambda(lambda img: cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)), 
        v2.Lambda(lambda img: cv2.split(img)[1]), # Extracting the green channel
        v2.Lambda(lambda img: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)),  
        v2.Lambda(lambda img: torch.tensor(img, dtype=torch.float32).unsqueeze(0)), # Adding a channel dimension 
        v2.Lambda(lambda img: img.expand(3, -1, -1)),  # Expanding to 3 channels
        v2.RandomCrop((256, 256)), 
        v2.RandomRotation((0, 360)),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
        return transform
    
    def augment_transform2(self):
        """Transform which only does preprocessing and random rotation"""
        transform = v2.Compose([
        v2.Resize((256, 256)),
        v2.Lambda(lambda img: img.convert("RGB")),
        v2.Lambda(lambda img: cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)),
        v2.Lambda(lambda img: cv2.split(img)[1]), # Extracting the green channel
        v2.Lambda(lambda img: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)),
        v2.Lambda(lambda img: torch.tensor(img, dtype=torch.float32).unsqueeze(0)), # Adding a channel dimension
        v2.Lambda(lambda img: img.expand(3, -1, -1)), # Expanding to 3 channels
        v2.RandomRotation((0, 360)),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
        return transform
    
    def test_transform(self):
        """Transform which only does preprocessing"""
        transform = v2.Compose([
        v2.Resize((256, 256)),
        v2.Lambda(lambda img: img.convert("RGB")),
        v2.Lambda(lambda img: cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)),
        v2.Lambda(lambda img: cv2.split(img)[1]), # Extracting the green channel
        v2.Lambda(lambda img: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)),
        v2.Lambda(lambda img: torch.tensor(img, dtype=torch.float32).unsqueeze(0)), # Adding a channel dimension
        v2.Lambda(lambda img: img.expand(3, -1, -1)), # Expanding to 3 channels
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
        return transform