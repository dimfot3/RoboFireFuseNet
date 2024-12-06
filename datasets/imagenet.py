import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


class ImageNet(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.classes = sorted(os.listdir(root_dir))  # Sort to ensure consistent ordering of classes
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            if os.path.isdir(cls_folder):
                for fname in os.listdir(cls_folder):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        self.image_paths.append(os.path.join(cls_folder, fname))
        self.image_paths = self.image_paths[::2]
        self.patch_size = 32
    def __len__(self):
        return len(self.image_paths)

    def gen_random_mask(self, x, mask_ratio):
        N = x.shape[0]
        L = (x.shape[2] // self.patch_size) ** 2
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.randn(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask

    def random_black_patches(self, image, patch_size=32, black_fraction=0.6):
            # Ensure image is in the expected format
        assert image.dim() == 3 and image.size(0) == 3, "Input image must be a 3xHxW tensor."
        
        # Get the height and width of the image
        C, H, W = image.size()
        
        # Calculate the number of patches along each dimension
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        
        # Create an array to hold the indices of patches
        patches_indices = np.arange(num_patches_h * num_patches_w)
        
        # Randomly select patches to blacken
        num_patches = num_patches_h * num_patches_w
        num_black_patches = int(np.ceil(num_patches * black_fraction))
        
        # Select random indices to blacken
        black_patch_indices = np.random.choice(patches_indices, num_black_patches, replace=False)
        mask = torch.ones_like(image)
        # Blacken the selected patches
        for index in black_patch_indices:
            patch_h = (index // num_patches_w) * patch_size
            patch_w = (index % num_patches_w) * patch_size
            # Blacken the selected patch
            image[:, patch_h:patch_h + patch_size, patch_w:patch_w + patch_size] = 0
            mask[:, patch_h:patch_h + patch_size, patch_w:patch_w + patch_size] = 0

        return image, mask

    def random_grayscale(self, image):
        gray_image = image.mean(dim=0, keepdim=True)  # Shape [1, H, W]
        gray_image = gray_image.repeat(3, 1, 1)  # Shape [3, H, W]
        gray_image[1:] *= 0
        return gray_image
    
    def roll_image(self, image, max_shift=32):
        shift_amount = np.random.randint(-max_shift, max_shift + 1)
        rolled_image = torch.roll(image, shifts=shift_amount, dims=-1)  # Shift along the width (W dimension)
        rolled_image = torch.roll(image, shifts=shift_amount, dims=-2)  # Shift along the width (W dimension)
        return rolled_image

    def detect_edges(self, image):
        # If the image is RGB (3 channels), convert to grayscale
        if image.size(0) == 3:
            image = 0.299 * image[0, :, :] + 0.587 * image[1, :, :] + 0.114 * image[2, :, :]
        
        # Add batch and channel dimensions for convolution
        image = image.unsqueeze(0).unsqueeze(0)

        # Define Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], 
                                [-2, 0, 2], 
                                [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], 
                                [ 0,  0,  0], 
                                [ 1,  2,  1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Convolve the image with Sobel filters
        grad_x = F.conv2d(image, sobel_x, padding=1)
        grad_y = F.conv2d(image, sobel_y, padding=1)

        # Compute edge magnitude
        edges = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        edges = (edges > edges.quantile(0.95)).float()
        return edges.squeeze()

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image).unsqueeze(0)
        label = torch.clone(image[0]).detach()
        images = torch.repeat_interleave(image, 2, dim=0)
        images[1] = self.roll_image(images[1], max_shift=32)
        images[1] = self.random_grayscale(images[1])
        images[1], _ = self.random_black_patches(images[1], patch_size=32, black_fraction=0.3)
        edges = self.detect_edges(images[0])
        images[0], mask = self.random_black_patches(images[0], patch_size=32, black_fraction=0.5)
        N, C, H, W = images.shape
        images = images.reshape(N * C, H, W)[:-2]
        mask = torch.ones_like(mask) - mask
        return images, label, mask, edges


if __name__ == '__main__':
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = ImageNet('../Datasets/FLAME2', transform_train)
    f, ax = plt.subplots(1, 4)
    imgs, label, mask, edges = dataset[0]
    img = np.transpose(imgs[:3].detach().cpu().numpy(), (1, 2, 0))* np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img2 = np.transpose(imgs[-1:].detach().cpu().numpy(), (1, 2, 0))* np.array([0.229, 0.224, 0.225]).mean() + np.array([0.485, 0.456, 0.406]).mean()
    ax[0].imshow(img)
    ax[1].imshow(img2)
    ax[2].imshow(edges)
    ax[-1].imshow(np.transpose(mask.detach().cpu().numpy(), (1, 2, 0)))
    plt.show()