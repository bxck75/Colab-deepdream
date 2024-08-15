import gradio as gr
import os
import numpy as np
import torch
import cv2 as cv
from enum import Enum

# Import necessary functions and classes from the original code
import utils.utils as utils
from utils.constants import *
import utils.video_utils as video_utils

# Assuming the original DeepDream functions are in a file called deepdream.py
import deepdream

class SupportedModels(Enum):
    VGG16_EXPERIMENTAL = "VGG16_EXPERIMENTAL"
    # Add other models as needed

class SupportedPretrainedWeights(Enum):
    IMAGENET = "IMAGENET"
    # Add other pretrained weights as needed

class SupportedTransforms(Enum):
    ZOOM_ROTATE = "ZOOM_ROTATE"
    # Add other transforms as needed

def run_deepdream_static(input_image, img_width, model, pretrained_weights, layers_to_use, pyramid_size, pyramid_ratio,
                         num_gradient_ascent_iterations, lr, spatial_shift_size, smoothing_coefficient, use_noise):
    config = {
        'img_width': img_width,
        'model': SupportedModels[model],
        'pretrained_weights': SupportedPretrainedWeights[pretrained_weights],
        'layers_to_use': layers_to_use.split(','),
        'pyramid_size': pyramid_size,
        'pyramid_ratio': pyramid_ratio,
        'num_gradient_ascent_iterations': num_gradient_ascent_iterations,
        'lr': lr,
        'spatial_shift_size': spatial_shift_size,
        'smoothing_coefficient': smoothing_coefficient,
        'use_noise': use_noise,
        'inputs_path': 'data/input',
        'out_images_path': 'data/out-images',
        'should_display': False
    }
    
    # Convert PIL Image to numpy array
    img_array = np.array(input_image)
    
    # Run DeepDream
    output_img = deepdream.deep_dream_static_image(config, img_array)
    
    return output_img

def run_deepdream_video(input_video, img_width, model, pretrained_weights, layers_to_use, pyramid_size, pyramid_ratio,
                        num_gradient_ascent_iterations, lr, spatial_shift_size, smoothing_coefficient, use_noise, blend):
    config = {
        'input': input_video,
        'img_width': img_width,
        'model': SupportedModels[model],
        'pretrained_weights': SupportedPretrainedWeights[pretrained_weights],
        'layers_to_use': layers_to_use.split(','),
        'pyramid_size': pyramid_size,
        'pyramid_ratio': pyramid_ratio,
        'num_gradient_ascent_iterations': num_gradient_ascent_iterations,
        'lr': lr,
        'spatial_shift_size': spatial_shift_size,
        'smoothing_coefficient': smoothing_coefficient,
        'use_noise': use_noise,
        'blend': blend,
        'inputs_path': 'data/input',
        'out_videos_path': 'data/out-videos',
        'should_display': False
    }
    
    # Run DeepDream video processing
    deepdream.deep_dream_video(config)
    
    # Return the path to the output video
    return os.path.join(config['out_videos_path'], f"deepdream_{os.path.basename(input_video)}")

def run_deepdream_ouroboros(input_image, img_width, model, pretrained_weights, layers_to_use, pyramid_size, pyramid_ratio,
                            num_gradient_ascent_iterations, lr, spatial_shift_size, smoothing_coefficient, use_noise,
                            video_length, frame_transform):
    config = {
        'img_width': img_width,
        'model': SupportedModels[model],
        'pretrained_weights': SupportedPretrainedWeights[pretrained_weights],
        'layers_to_use': layers_to_use.split(','),
        'pyramid_size': pyramid_size,
        'pyramid_ratio': pyramid_ratio,
        'num_gradient_ascent_iterations': num_gradient_ascent_iterations,
        'lr': lr,
        'spatial_shift_size': spatial_shift_size,
        'smoothing_coefficient': smoothing_coefficient,
        'use_noise': use_noise,
        'video_length': video_length,
        'frame_transform': SupportedTransforms[frame_transform],
        'inputs_path': 'data/input',
        'out_videos_path': 'data/out-videos',
        'should_display': False,
        'is_video': True
    }
    
    # Save the input image temporarily
    temp_input_path = os.path.join(config['inputs_path'], 'temp_input.jpg')
    input_image.save(temp_input_path)
    config['input'] = 'temp_input.jpg'
    
    # Run DeepDream video ouroboros
    deepdream.deep_dream_video_ouroboros(config)
    
    # Remove temporary input file
    os.remove(temp_input_path)
    
    # Return the path to the output video
    return os.path.join(config['out_videos_path'], "deepdream_ouroboros.mp4")

# Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# DeepDream Playground")
    
    with gr.Tab("Static Image"):
        with gr.Row():
            static_input = gr.Image(type="pil", label="Input Image")
            static_output = gr.Image(type="numpy", label="DeepDream Output")
        
        with gr.Row():
            static_img_width = gr.Slider(100, 1000, 600, step=1, label="Image Width")
            static_model = gr.Dropdown(["VGG16_EXPERIMENTAL"], label="Model")
            static_weights = gr.Dropdown(["IMAGENET"], label="Pretrained Weights")
            static_layers = gr.Textbox(value="relu4_3", label="Layers to Use (comma-separated)")
        
        with gr.Row():
            static_pyramid_size = gr.Slider(1, 10, 4, step=1, label="Pyramid Size")
            static_pyramid_ratio = gr.Slider(1.0, 2.0, 1.8, step=0.1, label="Pyramid Ratio")
            static_iterations = gr.Slider(1, 50, 10, step=1, label="Gradient Ascent Iterations")
            static_lr = gr.Slider(0.01, 1.0, 0.09, step=0.01, label="Learning Rate")
        
        with gr.Row():
            static_shift_size = gr.Slider(0, 100, 32, step=1, label="Spatial Shift Size")
            static_smoothing = gr.Slider(0.0, 2.0, 0.5, step=0.1, label="Smoothing Coefficient")
            static_use_noise = gr.Checkbox(label="Use Noise as Starting Point")
        
        static_button = gr.Button("Generate DeepDream")
    
    with gr.Tab("Video"):
        with gr.Row():
            video_input = gr.Video(label="Input Video")
            video_output = gr.Video(label="DeepDream Output")
        
        with gr.Row():
            video_img_width = gr.Slider(100, 1000, 600, step=1, label="Frame Width")
            video_model = gr.Dropdown(["VGG16_EXPERIMENTAL"], label="Model")
            video_weights = gr.Dropdown(["IMAGENET"], label="Pretrained Weights")
            video_layers = gr.Textbox(value="relu4_3", label="Layers to Use (comma-separated)")
        
        with gr.Row():
            video_pyramid_size = gr.Slider(1, 10, 4, step=1, label="Pyramid Size")
            video_pyramid_ratio = gr.Slider(1.0, 2.0, 1.8, step=0.1, label="Pyramid Ratio")
            video_iterations = gr.Slider(1, 50, 10, step=1, label="Gradient Ascent Iterations")
            video_lr = gr.Slider(0.01, 1.0, 0.09, step=0.01, label="Learning Rate")
        
        with gr.Row():
            video_shift_size = gr.Slider(0, 100, 32, step=1, label="Spatial Shift Size")
            video_smoothing = gr.Slider(0.0, 2.0, 0.5, step=0.1, label="Smoothing Coefficient")
            video_use_noise = gr.Checkbox(label="Use Noise as Starting Point")
            video_blend = gr.Slider(0.0, 1.0, 0.85, step=0.05, label="Blend Coefficient")
        
        video_button = gr.Button("Generate DeepDream Video")
    
    with gr.Tab("Ouroboros"):
        with gr.Row():
            ouroboros_input = gr.Image(type="pil", label="Input Image")
            ouroboros_output = gr.Video(label="DeepDream Ouroboros Output")
        
        with gr.Row():
            ouroboros_img_width = gr.Slider(100, 1000, 600, step=1, label="Frame Width")
            ouroboros_model = gr.Dropdown(["VGG16_EXPERIMENTAL"], label="Model")
            ouroboros_weights = gr.Dropdown(["IMAGENET"], label="Pretrained Weights")
            ouroboros_layers = gr.Textbox(value="relu4_3", label="Layers to Use (comma-separated)")
        
        with gr.Row():
            ouroboros_pyramid_size = gr.Slider(1, 10, 4, step=1, label="Pyramid Size")
            ouroboros_pyramid_ratio = gr.Slider(1.0, 2.0, 1.8, step=0.1, label="Pyramid Ratio")
            ouroboros_iterations = gr.Slider(1, 50, 10, step=1, label="Gradient Ascent Iterations")
            ouroboros_lr = gr.Slider(0.01, 1.0, 0.09, step=0.01, label="Learning Rate")
        
        with gr.Row():
            ouroboros_shift_size = gr.Slider(0, 100, 32, step=1, label="Spatial Shift Size")
            ouroboros_smoothing = gr.Slider(0.0, 2.0, 0.5, step=0.1, label="Smoothing Coefficient")
            ouroboros_use_noise = gr.Checkbox(label="Use Noise as Starting Point")
            ouroboros_video_length = gr.Slider(1, 100, 30, step=1, label="Video Length (frames)")
            ouroboros_transform = gr.Dropdown(["ZOOM_ROTATE"], label="Frame Transform")
        
        ouroboros_button = gr.Button("Generate DeepDream Ouroboros")

    static_button.click(
        run_deepdream_static,
        inputs=[static_input, static_img_width, static_model, static_weights, static_layers, static_pyramid_size,
                static_pyramid_ratio, static_iterations, static_lr, static_shift_size, static_smoothing, static_use_noise],
        outputs=static_output
    )

    video_button.click(
        run_deepdream_video,
        inputs=[video_input, video_img_width, video_model, video_weights, video_layers, video_pyramid_size,
                video_pyramid_ratio, video_iterations, video_lr, video_shift_size, video_smoothing, video_use_noise, video_blend],
        outputs=video_output
    )

    ouroboros_button.click(
        run_deepdream_ouroboros,
        inputs=[ouroboros_input, ouroboros_img_width, ouroboros_model, ouroboros_weights, ouroboros_layers, ouroboros_pyramid_size,
                ouroboros_pyramid_ratio, ouroboros_iterations, ouroboros_lr, ouroboros_shift_size, ouroboros_smoothing,
                ouroboros_use_noise, ouroboros_video_length, ouroboros_transform],
        outputs=ouroboros_output
    )

iface.launch()
