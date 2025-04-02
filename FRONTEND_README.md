# Music Mixing Style Transfer - Frontend Tools

This repository provides tools to demonstrate and utilize the Music Mixing Style Transfer capability introduced in the paper [*"Music Mixing Style Transfer: A Contrastive Learning Approach to Disentangle Audio Effects"*](https://arxiv.org/abs/2211.02247) by Junghyun Koo, Marco A. Martínez-Ramírez, Wei-Hsiang Liao, Stefan Uhlich, Kyogu Lee, and Yuki Mitsufuji.

## What is Music Mixing Style Transfer?

This technology allows you to transfer the mixing style (audio effects, EQ, compression, etc.) from a reference track to your input track. It's like applying Instagram filters, but for audio!

## Getting Started

### Prerequisites

1. Make sure you have all the required dependencies installed:

```bash
pip install -r requirements.txt
```

2. Download the pre-trained model weights from the links in the main README.md and place them in the `weights` folder:
   - [FXencoder (Φp.s.)](https://drive.google.com/file/d/1BFABsJRUVgJS5UE5iuM03dbfBjmI9LT5/view?usp=sharing)
   - [MixFXcloner](https://drive.google.com/file/d/1Qu8rD7HpTNA1gJUVp2IuaeU_Nue8-VA3/view?usp=sharing)

### Available Tools

We provide two ways to use the Music Mixing Style Transfer capability:

1. **Web Interface**: An interactive web app for processing individual files
2. **Batch Processing**: A command-line tool for processing multiple files

## Web Interface

The web interface provides an easy-to-use graphical interface for processing individual audio files.

### Running the Web Interface

To start the web interface, run:

```bash
python app.py
```

A local web server will start, and the interface will be available at http://localhost:7860.

### Using the Web Interface

1. **Upload Input Audio**: This is the track you want to modify
2. **Upload Reference Audio**: This is the track with the mixing style you want to apply to your input
3. **Configure Options**:
   - **Use Interpolation**: Enable for a more gradual style transfer
   - **Normalize Input**: Enable audio effects normalization (recommended)
   - **Separation Model**: Choose between `mdx_extra` (default) or `htdemucs`
4. **Process Style Transfer**: Click this button to start the processing

For more details, see [interface_README.md](interface_README.md).

## Batch Processing

The batch processing tool allows you to process multiple audio files with the same reference style.

### Running Batch Processing

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

For more details and examples, see [batch_process_README.md](batch_process_README.md).

## How It Works

The Music Mixing Style Transfer system works through the following process:

1. Source Separation: Both input and reference tracks are separated into four stems (drums, bass, vocals, other)
2. Style Extraction: The FXencoder model extracts mixing style features from the reference stems
3. Style Application: The MixFXcloner model applies these style features to the input stems
4. Remixing: The processed stems are combined to create the final output

## Tips for Best Results

- Use WAV files with stereo channels, 44.1kHz sampling rate, and 16-bit depth
- Input tracks that aren't too loud generally work better
- Processing may take some time, especially for longer tracks
- For more control, try processing with and without normalization
- If stems aren't separating well, try the alternative separation model

## Troubleshooting

- If you encounter errors, make sure all model weights are downloaded and placed in the `weights` folder
- Check that all dependencies are installed correctly
- Make sure your audio files are properly formatted (stereo, 44.1kHz, WAV format)
- If processing fails, try with shorter audio clips first to verify everything is working

## Citation

If you use this tool in your work, please consider citing the original paper:

```
@article{koo2022music,
  title={Music Mixing Style Transfer: A Contrastive Learning Approach to Disentangle Audio Effects},
  author={Koo, Junghyun and Martinez-Ramirez, Marco A and Liao, Wei-Hsiang and Uhlich, Stefan and Lee, Kyogu and Mitsufuji, Yuki},
  journal={arXiv preprint arXiv:2211.02247},
  year={2022}
}
``` 