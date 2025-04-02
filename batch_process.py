#!/usr/bin/env python
"""
Batch processing script for Music Mixing Style Transfer.
This script allows you to process multiple input files with a single reference style.
"""

import os
import sys
import argparse
import torch
import yaml
import shutil
import time
from pathlib import Path

# Add the project modules to path
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(currentdir, "mixing_style_transfer"))
sys.path.append(os.path.join(currentdir, "inference"))

# Import necessary modules
from inference.style_transfer import Mixing_Style_Transfer_Inference

# Setup configuration
default_ckpt_path_enc = os.path.join('weights', 'FXencoder_ps.pt')
default_ckpt_path_conv = os.path.join('weights', 'MixFXcloner_ps.pt')
default_norm_feature_path = os.path.join('weights', 'musdb18_fxfeatures_eqcompimagegain.npy')

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
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Batch processing for Music Mixing Style Transfer')
    
    # Input and output
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Directory containing input audio files to process')
    parser.add_argument('--reference_file', type=str, required=True, 
                        help='Reference audio file to use for style transfer')
    parser.add_argument('--output_dir', type=str, default='./batch_output',
                        help='Directory to save processed audio files')
    
    # Processing options
    parser.add_argument('--interpolation', type=str2bool, default=False,
                        help='Enable interpolation for gradual style transfer')
    parser.add_argument('--normalize_input', type=str2bool, default=True,
                        help='Apply audio effects normalization')
    parser.add_argument('--separation_model', type=str, default='mdx_extra',
                        choices=['mdx_extra', 'htdemucs'],
                        help='Model to use for source separation')
    
    # Model paths
    parser.add_argument('--ckpt_path_enc', type=str, default=default_ckpt_path_enc,
                        help='Path to FXencoder checkpoint')
    parser.add_argument('--ckpt_path_conv', type=str, default=default_ckpt_path_conv,
                        help='Path to MixFXcloner checkpoint')
    parser.add_argument('--norm_feature_path', type=str, default=default_norm_feature_path,
                        help='Path to normalization features')
    
    # Device options
    parser.add_argument('--inference_device', type=str, default='gpu',
                        choices=['cpu', 'gpu'],
                        help='Device to use for inference')
    parser.add_argument('--separation_device', type=str, default='cpu',
                        choices=['cpu', 'gpu'],
                        help='Device to use for source separation')
    
    return parser.parse_args()

def prepare_temp_directory(input_file, reference_file, temp_dir):
    """Prepare temporary directory for processing"""
    song_dir = os.path.join(temp_dir, os.path.basename(input_file).split('.')[0])
    os.makedirs(song_dir, exist_ok=True)
    
    # Copy input and reference files to temp directory
    input_dest = os.path.join(song_dir, "input.wav")
    ref_dest = os.path.join(song_dir, "reference.wav")
    
    shutil.copy2(input_file, input_dest)
    shutil.copy2(reference_file, ref_dest)
    
    return temp_dir

def create_args_object(args, temp_dir):
    """Create argument object for the style transfer model"""
    class Args:
        pass
    
    obj = Args()
    
    # Directory paths
    obj.target_dir = temp_dir + '/'
    obj.output_dir = temp_dir + '/'
    obj.input_file_name = 'input'
    obj.reference_file_name = 'reference'
    obj.reference_file_name_2interpolate = 'reference_B'
    
    # Model weights
    obj.ckpt_path_enc = args.ckpt_path_enc
    obj.ckpt_path_conv = args.ckpt_path_conv
    obj.precomputed_normalization_feature = args.norm_feature_path
    
    # Inference args
    obj.sample_rate = 44100
    obj.segment_length = 2**19
    obj.segment_length_ref = 2**19
    obj.instruments = ["drums", "bass", "other", "vocals"]
    obj.stem_level_directory_name = 'separated'
    obj.save_each_inst = False
    obj.do_not_separate = False
    obj.separation_model = args.separation_model
    obj.normalize_input = args.normalize_input
    obj.normalization_order = ['loudness', 'eq', 'compression', 'imager', 'loudness']
    obj.interpolation = args.interpolation
    obj.interpolate_segments = 30
    
    # Device args
    obj.workers = 1
    obj.inference_device = args.inference_device
    obj.batch_size = 1
    obj.separation_device = args.separation_device
    
    # Load network configurations
    configs = load_config()
    obj.cfg_encoder = configs['Effects_Encoder']['default']
    obj.cfg_converter = configs['TCN']['default']
    
    return obj

def process_file(input_file, reference_file, args):
    """Process a single file with the style transfer model"""
    print(f"Processing: {os.path.basename(input_file)}")
    
    # Create temporary directory
    temp_dir = os.path.join('./temp_data', f"batch_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Prepare directory with input and reference files
        prepare_temp_directory(input_file, reference_file, temp_dir)
        
        # Create args object
        model_args = create_args_object(args, temp_dir)
        
        # Initialize the style transfer model
        print("  Initializing style transfer model...")
        inference_style_transfer = Mixing_Style_Transfer_Inference(model_args)
        
        # Run inference
        print("  Running style transfer...")
        if model_args.interpolation:
            inference_style_transfer.inference_interpolation()
        else:
            inference_style_transfer.inference()
        
        # Get result file
        output_name_tag = 'output' if model_args.normalize_input else 'output_notnormed'
        if model_args.interpolation:
            output_name_tag = 'output_interpolation' if model_args.normalize_input else 'output_notnormed_interpolation'
        
        song_dir = os.path.join(temp_dir, os.path.basename(input_file).split('.')[0])
        output_file = os.path.join(song_dir, f"mixture_{output_name_tag}.wav")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Copy result to output directory
        output_path = os.path.join(
            args.output_dir, 
            f"{os.path.splitext(os.path.basename(input_file))[0]}_styled.wav"
        )
        
        if os.path.exists(output_file):
            shutil.copy2(output_file, output_path)
            print(f"  Saved output to: {output_path}")
            return True
        else:
            print(f"  Error: Output file not generated for {input_file}")
            return False
    
    except Exception as e:
        print(f"  Error processing {input_file}: {str(e)}")
        return False
    
    finally:
        # Clean up temporary files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def main():
    """Main function for batch processing"""
    args = parse_args()
    
    # Validate reference file
    if not os.path.isfile(args.reference_file):
        print(f"Error: Reference file does not exist: {args.reference_file}")
        return
    
    # Find input files
    if os.path.isdir(args.input_dir):
        input_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) 
                      if f.lower().endswith(('.wav', '.WAV'))]
    else:
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    if not input_files:
        print(f"Error: No WAV files found in input directory: {args.input_dir}")
        return
    
    print(f"Found {len(input_files)} input files to process")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process files
    success_count = 0
    for i, input_file in enumerate(input_files):
        print(f"\n[{i+1}/{len(input_files)}] Processing {os.path.basename(input_file)}")
        if process_file(input_file, args.reference_file, args):
            success_count += 1
    
    print(f"\nProcessing complete: {success_count}/{len(input_files)} files processed successfully")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    # Ensure temp directory exists
    os.makedirs('./temp_data', exist_ok=True)
    
    # Run batch processing
    main()
    
    # Final cleanup
    if os.path.exists('./temp_data'):
        shutil.rmtree('./temp_data') 