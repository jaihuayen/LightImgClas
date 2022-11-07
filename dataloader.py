import torchvision.transforms as transforms

from dataset import LightImgDataset

def get_loaders(args,
                img_mean=(0.485, 0.456, 0.406),
                img_std=(0.229, 0.224, 0.225),
                img_size=(224, 224)):

    # Image Transformation for data-preprocessing

    transform = transforms.Compose([transforms.Resize(img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(img_mean, img_std)])

    