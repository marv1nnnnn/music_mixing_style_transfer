import os
import sys
import gradio as gr
import torch
import numpy as np
import yaml
import tempfile
import soundfile as sf
import time
import shutil
from pathlib import Path

# Add the project modules to path
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(currentdir, "mixing_style_transfer"))
sys.path.append(os.path.join(currentdir, "inference"))

# Import necessary modules
from inference.style_transfer import Mixing_Style_Transfer_Inference
from networks import FXencoder, TCNModel

# Setup configuration
default_ckpt_path_enc = os.path.join('weights', 'FXencoder_ps.pt')
default_ckpt_path_conv = os.path.join('weights', 'MixFXcloner_ps.pt')
default_norm_feature_path = os.path.join('weights', 'musdb18_fxfeatures_eqcompimagegain.npy')

class Args:
    """Argument class to store configuration"""
    pass

def load_config():
    """Load configuration from yaml file"""
    with open(os.path.join(currentdir, 'inference', 'configs.yaml'), 'r') as f:
        configs = yaml.full_load(f)
    return configs

def str2bool(v):
    """Convert string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

def create_args(normalize_input=True, separation_model="mdx_extra"):
    """Create arguments object with default values"""
    args = Args()
    
    # directory paths
    args.target_dir = os.path.join(os.getcwd(), 'temp_data')
    args.output_dir = os.path.join(os.getcwd(), 'temp_data')
    args.input_file_name = 'input'
    args.reference_file_name = 'reference'
    args.reference_file_name_2interpolate = 'reference_B'
    
    # saved weights
    args.ckpt_path_enc = os.path.abspath(default_ckpt_path_enc)
    args.ckpt_path_conv = os.path.abspath(default_ckpt_path_conv)
    args.precomputed_normalization_feature = os.path.abspath(default_norm_feature_path)
    
    # inference args
    args.sample_rate = 44100
    args.segment_length = 2**19
    args.segment_length_ref = 2**19
    args.instruments = ["drums", "bass", "other", "vocals"]
    args.stem_level_directory_name = 'separated'
    args.save_each_inst = False
    args.do_not_separate = False
    args.separation_model = separation_model
    args.normalize_input = normalize_input
    args.normalization_order = ['loudness', 'eq', 'compression', 'imager', 'loudness']
    args.interpolation = False
    args.interpolate_segments = 30
    
    # device args
    args.workers = 1
    args.inference_device = 'gpu'
    args.batch_size = 1
    args.separation_device = 'cpu'
    
    # load network configurations
    configs = load_config()
    args.cfg_encoder = configs['Effects_Encoder']['default']
    args.cfg_converter = configs['TCN']['default']
    
    return args

def prepare_data_directory(input_audio, reference_audio):
    """Prepare data directory for processing"""
    # Create temp directories
    temp_dir = os.path.join(os.getcwd(), 'temp_data', f"temp_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save input and reference files
    input_path = os.path.join(temp_dir, "input.wav")
    reference_path = os.path.join(temp_dir, "reference.wav")
    
    # Handle audio data format from Gradio
    # Gradio returns a tuple of (sample_rate, data) for audio inputs
    input_data = input_audio[1] if isinstance(input_audio[1], np.ndarray) else input_audio[1].numpy()
    reference_data = reference_audio[1] if isinstance(reference_audio[1], np.ndarray) else reference_audio[1].numpy()
    
    # Ensure the data is in the correct format (samples, channels)
    if input_data.ndim == 1:
        input_data = input_data.reshape(-1, 1)
    if reference_data.ndim == 1:
        reference_data = reference_data.reshape(-1, 1)
    
    # If mono, duplicate to stereo
    if input_data.shape[1] == 1:
        input_data = np.tile(input_data, (1, 2))
    if reference_data.shape[1] == 1:
        reference_data = np.tile(reference_data, (1, 2))
    
    # Transpose if necessary (soundfile expects (samples, channels))
    if input_data.shape[0] == 2:
        input_data = input_data.T
    if reference_data.shape[0] == 2:
        reference_data = reference_data.T
    
    # Save the files
    sf.write(input_path, input_data, input_audio[0], 'PCM_16')
    sf.write(reference_path, reference_data, reference_audio[0], 'PCM_16')
    
    return temp_dir

def process_style_transfer(input_audio, reference_audio, use_interpolation=False, normalize_input=True, separation_model="mdx_extra", progress=gr.Progress()):
    """Process the style transfer on input and reference audio"""
    if input_audio is None or reference_audio is None:
        return None, "Error: Please upload both input and reference audio files."
    
    try:
        # Create args with default configuration
        args = create_args(normalize_input=normalize_input, separation_model=separation_model)
        
        # Set interpolation if needed
        args.interpolation = use_interpolation
        
        # Prepare data directory
        temp_dir = prepare_data_directory(input_audio, reference_audio)
        args.target_dir = temp_dir
        args.output_dir = temp_dir
        
        progress(0.1, "Preparing audio files")
        
        # Initialize the style transfer model
        progress(0.2, "Initializing style transfer model")
        inference_style_transfer = Mixing_Style_Transfer_Inference(args)
        
        # Run inference
        progress(0.4, "Running style transfer")
        if args.interpolation:
            inference_style_transfer.inference_interpolation()
        else:
            inference_style_transfer.inference()
        
        progress(0.9, "Processing complete")
        
        # Get result file
        output_name_tag = 'output' if args.normalize_input else 'output_notnormed'
        if args.interpolation:
            output_name_tag = 'output_interpolation' if args.normalize_input else 'output_notnormed_interpolation'
        
        output_file = os.path.join(temp_dir, f"mixture_{output_name_tag}.wav")
        
        if os.path.exists(output_file):
            # Load the output file
            data, samplerate = sf.read(output_file)
            progress(1.0, "Done")
            # Clean up temporary files
            cleanup_temp_files()
            return (samplerate, data), "Processing completed successfully!"
        else:
            progress(1.0, "Error")
            return None, "Error: Style transfer processing failed. Output file not generated."
            
    except Exception as e:
        import traceback
        error_msg = f"Error during processing: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, f"Error: {str(e)}"

def cleanup_temp_files():
    """Clean up temporary files and directories"""
    if os.path.exists('./temp_data'):
        shutil.rmtree('./temp_data')

# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="Music Mixing Style Transfer Demo", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Music Mixing Style Transfer Demo")
        gr.Markdown("""This demo shows the music mixing style transfer capability introduced in the paper:
                     **"Music Mixing Style Transfer: A Contrastive Learning Approach to Disentangle Audio Effects"**
                     by Junghyun Koo, Marco A. Martínez-Ramírez, Wei-Hsiang Liao, Stefan Uhlich, Kyogu Lee, and Yuki Mitsufuji.
                     
                     Upload an input audio file and a reference audio file to transfer the mixing style from the reference to the input.""")
        
        with gr.Row():
            with gr.Column():
                input_audio = gr.Audio(label="Input Audio", type="numpy")
                reference_audio = gr.Audio(label="Reference Audio", type="numpy")
                
                with gr.Row():
                    with gr.Column():
                        use_interpolation = gr.Checkbox(label="Use Interpolation", value=False)
                        normalize_input = gr.Checkbox(label="Normalize Input", value=True, 
                                                     info="Enable audio effects normalization (recommended)")
                    with gr.Column():
                        separation_model = gr.Radio(
                            ["mdx_extra", "htdemucs"], 
                            label="Separation Model", 
                            value="mdx_extra",
                            info="Model used to separate stems from the audio"
                        )
                
                process_btn = gr.Button("Process Style Transfer", variant="primary")
                status_text = gr.Textbox(label="Status", value="Ready", interactive=False)
                
            with gr.Column():
                output_audio = gr.Audio(label="Output Audio", type="numpy")
                
        # Set up processing on button click
        process_btn.click(
            fn=process_style_transfer,
            inputs=[input_audio, reference_audio, use_interpolation, normalize_input, separation_model],
            outputs=[output_audio, status_text]
        )
        
        # Info section
        with gr.Accordion("How it works", open=False):
            gr.Markdown("""
            ## Process
            
            1. The system separates both tracks into four stems: drums, bass, vocals, and other instruments
            2. The FXencoder extracts audio effect embeddings from the reference track
            3. The MixFXcloner applies these effects to the input track
            4. The processed stems are remixed to create the output track
            
            ## Advanced Options
            
            - **Use Interpolation**: Creates a gradual style transfer effect
            - **Normalize Input**: Applies audio effects normalization to better isolate style (recommended)
            - **Separation Model**: 
              - `mdx_extra`: Default model with good overall performance
              - `htdemucs`: Alternative model that may work better for some tracks
            """)
        
        # Tips section
        with gr.Accordion("Tips for best results", open=False):
            gr.Markdown("""
            - Use WAV files with stereo channels, 44.1kHz sampling rate, and 16-bit depth
            - Input tracks that aren't too loud generally work better
            - Processing may take some time, especially for longer tracks
            - For more control, try processing with and without normalization
            - If stems aren't separating well, try the alternative separation model
            """)
    
    return app

# Main function
if __name__ == "__main__":
    # Ensure temp directory exists
    os.makedirs('./temp_data', exist_ok=True)
    
    # Create and launch the interface
    app = create_interface()
    app.launch(share=True)
    
    # Clean up temporary files on exit
    cleanup_temp_files() 