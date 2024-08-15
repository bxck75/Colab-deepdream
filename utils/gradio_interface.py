import gradio as gr
import os
import numpy as np
import torch
from torchvision import transforms
import cv2

# Import necessary functions and classes from the original code
from utils.constants import *
import utils.utils as utils
from models.definitions.vggs import Vgg16

def deep_dream_simple(img, n_iterations, learning_rate):
    # Normalize image
    img = (img - IMAGENET_MEAN_1) / IMAGENET_STD_1
    
    # Transform into PyTorch tensor
    img_tensor = transforms.ToTensor()(img).to(DEVICE).unsqueeze(0)
    img_tensor.requires_grad = True

    model = Vgg16(requires_grad=False).to(DEVICE)

    for _ in range(n_iterations):
        out = model(img_tensor)
        activations = out.relu4_3
        activations.backward(activations)

        img_tensor_grad = img_tensor.grad.data
        smooth_grads = img_tensor_grad / torch.std(img_tensor_grad)
        img_tensor.data += learning_rate * smooth_grads

        img_tensor.grad.data.zero_()

    # Convert back to numpy array
    img = np.moveaxis(img_tensor.to('cpu').detach().numpy()[0], 0, 2)
    img = (img * IMAGENET_STD_1) + IMAGENET_MEAN_1
    img = (np.clip(img, 0., 1.) * 255).astype(np.uint8)

    return img

def run_deepdream(input_image, n_iterations, learning_rate):
    # Convert PIL Image to numpy array
    img_array = np.array(input_image)
    
    # Run DeepDream
    output_img = deep_dream_simple(img_array, n_iterations, learning_rate)
    
    return output_img

# Gradio interface
iface = gr.Interface(
    fn=run_deepdream,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Slider(1, 50, 10, step=1, label="Number of Iterations"),
        gr.Slider(0.1, 1.0, 0.3, step=0.1, label="Learning Rate")
    ],
    outputs=gr.Image(type="numpy", label="DeepDream Output"),
    title="DeepDream Playground",
    description="Upload an image and adjust parameters to see the DeepDream effect."
)

if __name__ == "__main__":
    iface.launch()
