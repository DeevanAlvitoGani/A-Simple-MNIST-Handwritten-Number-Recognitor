import torch
from torchvision import datasets
from torchvision.transforms import ToPILImage, ToTensor
import random

# Load the MNIST dataset
mnist_data = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())

# Pick a random index from the dataset
random_index = random.randint(0, len(mnist_data) - 1)

# Get the random image and label
image_tensor, label = mnist_data[random_index]

# Convert the tensor to a PIL image
to_pil = ToPILImage()
image = to_pil(image_tensor)

# Save the image locally
image_filename = f"img{label}.png"
image.save(image_filename)

print(f"Saved random MNIST image with label {label} as '{image_filename}'")
