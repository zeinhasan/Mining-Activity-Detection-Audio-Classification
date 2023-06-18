import os
import librosa
import numpy as np
import soundfile as sf
import pydub
# Input and output folder paths
input_folder_No = "Dataset/No"
output_folder_No = "Dataset/No/Augmented"
# Augmentation parameters
replication_factor = 10
desired_duration = 15
# Create the output folder if it doesn't exist
if not os.path.exists(output_folder_No):
    os.makedirs(output_folder_No)
    # Loop over each audio file in the input folder
for filename in os.listdir(input_folder_No):
    if filename.endswith('.mp3'):
        # Load audio file using pydub
        audio_file = os.path.join(input_folder_No, filename)
        audio = pydub.AudioSegment.from_mp3(audio_file)
        audio = np.array(audio.get_array_of_samples())
        sr = audio.frame_rate

        # Adjust audio duration to desired duration
        target_length = int(desired_duration * sr)
        if len(audio) < target_length:
            # Pad with zeros if audio is shorter
            audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
        elif len(audio) > target_length:
            # Trim if audio is longer
            audio = audio[:target_length]

        # Augmentation and replication
        augmented_files = []
        augmented_files.append(('stretched', pydub.effects.time_stretch(audio.astype(np.int16), 0.8)))
        augmented_files.append(('pitch_shifted', pydub.effects.pitch_shift(audio.astype(np.int16), sr, 2)))
        augmented_files.append(('speed_changed', pydub.effects.speedup(audio.astype(np.int16), playback_speed=1.2)))
        augmented_files.append(('spec_augment', audio))  # No spectrogram-based augmentation for MP3 files

        # Generate the output files
        for augmentation, augmented_audio in augmented_files:
            for i in range(replication_factor):
                # Generate the output file path
                output_filename = f"{filename.replace('.mp3', '')}_{augmentation}_{i}.wav"
                output_file = os.path.join(output_folder_No, output_filename)

                # Save augmented audio
                sf.write(output_file, augmented_audio, sr, subtype='PCM_16')