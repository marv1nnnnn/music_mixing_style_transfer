# Music Mixing Style Transfer Web Interface

This is a web interface for the "Music Mixing Style Transfer" project, allowing you to transfer the mixing style from a reference audio track to your input track.

## Installation

1. Make sure you have all the required dependencies installed:

```bash
pip install -r requirements.txt
```

2. Download the pre-trained model weights from the links in the main README.md and place them in the `weights` folder:
   - [FXencoder (Φp.s.)](https://drive.google.com/file/d/1BFABsJRUVgJS5UE5iuM03dbfBjmI9LT5/view?usp=sharing)
   - [MixFXcloner](https://drive.google.com/file/d/1Qu8rD7HpTNA1gJUVp2IuaeU_Nue8-VA3/view?usp=sharing)

## Running the Web Interface

To start the web interface, run:

```bash
python app.py
```

A local web server will start, and the interface will be available at http://localhost:7860.

## Using the Interface

1. **Upload Input Audio**: This is the track you want to modify
2. **Upload Reference Audio**: This is the track with the mixing style you want to apply to your input
3. **Configure Options**:
   - **Use Interpolation**: Enable for a more gradual style transfer
   - **Normalize Input**: Enable audio effects normalization (recommended)
   - **Separation Model**: Choose between `mdx_extra` (default) or `htdemucs`
4. **Process Style Transfer**: Click this button to start the processing

The output will appear on the right side of the interface. You can play it directly in the browser or download it.

## Advanced Options

### Normalization

The "Normalize Input" option applies audio effects normalization to better isolate the style characteristics. It's recommended to keep this enabled for most cases, but you can experiment by disabling it to see if you prefer the results.

### Separation Models

The system uses source separation to split tracks into different stems before applying style transfer:

- **mdx_extra**: Default model with good overall performance
- **htdemucs**: Alternative model that may work better for some tracks

If you're not satisfied with the separation quality, try switching between these models.

### Interpolation

The "Use Interpolation" option creates a gradual style transfer effect. This can be useful for more subtle transitions between the original and the styled track.

## How It Works

1. The system separates both tracks into four stems: drums, bass, vocals, and other instruments
2. The FXencoder extracts audio effect embeddings from the reference track
3. The MixFXcloner applies these effects to the input track
4. The processed stems are remixed to create the output track

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