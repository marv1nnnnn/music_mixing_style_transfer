# Music Mixing Style Transfer Batch Processing

This tool allows you to process multiple audio files with the same reference style in a batch operation.

## Usage

```bash
python batch_process.py --input_dir INPUT_DIRECTORY --reference_file REFERENCE_FILE [options]
```

### Required Arguments

- `--input_dir`: Directory containing input WAV files to process
- `--reference_file`: Path to the reference WAV file to use for style transfer

### Optional Arguments

- `--output_dir`: Directory to save processed audio files (default: './batch_output')
- `--interpolation`: Enable interpolation for gradual style transfer (default: False)
- `--normalize_input`: Apply audio effects normalization (default: True)
- `--separation_model`: Model to use for source separation ('mdx_extra' or 'htdemucs', default: 'mdx_extra')
- `--inference_device`: Device to use for inference ('cpu' or 'gpu', default: 'gpu')
- `--separation_device`: Device to use for source separation ('cpu' or 'gpu', default: 'cpu')

## Examples

### Basic Usage

Process all WAV files in the 'input_songs' directory using 'reference.wav' as the reference style:

```bash
python batch_process.py --input_dir ./input_songs --reference_file ./reference.wav
```

### Advanced Usage

Process with interpolation and a specific separation model:

```bash
python batch_process.py --input_dir ./input_songs --reference_file ./reference.wav --interpolation true --separation_model htdemucs --output_dir ./styled_output
```

Process without normalization and using CPU for inference:

```bash
python batch_process.py --input_dir ./input_songs --reference_file ./reference.wav --normalize_input false --inference_device cpu
```

## Output

Processed files will be saved in the specified output directory (default: './batch_output') with the naming format:

```
original_filename_styled.wav
```

## Tips

- Make sure all input files are stereo WAV files at 44.1kHz
- If processing many files, consider setting both `--inference_device` and `--separation_device` to 'cpu' if you're experiencing memory issues
- For large audio files, processing may take some time
- Experiment with different separation models if you're not satisfied with the results

## Troubleshooting

- If you receive errors about missing model weights, make sure you have downloaded the required models and placed them in the 'weights' directory
- For memory issues, try processing smaller audio files or using the CPU for inference
- If source separation is failing, try using a different separation model 