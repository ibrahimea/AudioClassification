import torch
import torchaudio
import matplotlib.pyplot as plt
import os


# Load the audio file
audio_path = 'Data'
output_path='OutPut'
def Convert_Audio_Image(audio_path,output_path):
    for folder in os.listdir(audio_path):
        folder_path = os.path.join(audio_path,folder)
        for file in os.listdir(folder_path):
            waveform, sample_rate = torchaudio.load(os.path.join(folder_path, file))
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=128,  # Number of Mel bins
                n_fft=1024  # Increase n_fft for more frequency bins if needed
            )
            spectrogram = mel_spectrogram(waveform)

            # Convert the amplitude spectrogram to decibels for better visualization
            db_transform = torchaudio.transforms.AmplitudeToDB()
            spectrogram_db = db_transform(spectrogram)

            # Remove channel dimension (now should be (128, T))
            spectrogram_db = spectrogram_db.squeeze()
            name_class=folder_path.split('/')[-1]
            output_path_class=os.path.join(output_path,name_class)

            if not os.path.exists(output_path_class):
                os.mkdir(output_path_class)
            output_file = os.path.join(str(output_path_class), os.path.splitext(file)[0] + '.png')
            plt.imsave(output_file, spectrogram_db.numpy(), cmap='viridis')
Convert_Audio_Image(audio_path,output_path)


