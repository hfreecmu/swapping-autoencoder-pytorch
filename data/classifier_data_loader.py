from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import PIL.Image as Image

class ClassifierDataSet(Dataset):
    """Load images under folders"""
    def __init__(self, labels, transform, gan_augment, return_path):
        self.transform = transform
        self.labels = labels
        self.return_path = return_path
        self.gan_augment = gan_augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_loc = self.labels[idx]
        image = Image.open(img_loc[0]).convert("RGB")
        tensor_image = self.transform(image)

        if not self.return_path:
            return tensor_image, img_loc[1]
        else:
            return tensor_image, img_loc[1], img_loc[0]

def get_data_loader(labels, is_train, image_size=224, batch_size=16, gan_augment=False, return_path=False):
    #TODO do I want to add random cropping?
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size), Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size), Image.BICUBIC),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    dataset = ClassifierDataSet(labels, transform, gan_augment=gan_augment, return_path=return_path)
    
    dloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_train)

    return dloader