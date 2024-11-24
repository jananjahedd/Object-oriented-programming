import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.getcwd() + "/src/")
from src.dataorganisation import (ImageDataset, AudioDataset,
                                  TaskType, LoadMethod)
from src.batchloader import BatchLoader
from src.preprocessing import (PreprocessingPipeline,
                               CenterCrop, RandomCrop,
                               RandomCropAudio, PitchShift)


base_path = os.path.dirname(__file__)


def simulate_imagedata():
    # Values used for the image dataset
    image_root = os.path.join(base_path, "ImageData",
                              "images", "content", "images")
    image_task_type = "REGRESSION"
    image_load_method = "EAGER"
    image_labels_csv = os.path.join(base_path, "ImageData",
                                    "targets.csv")
    return (image_root, image_task_type,
            image_load_method, image_labels_csv)


def simulate_audiodata():
    # Values used for the audio dataset
    audio_root = os.path.join(base_path, "AudioData",
                              "genres_original")
    audio_task_type = "CLASSIFICATION"
    audio_load_method = "LAZY"
    audio_labels_csv = os.path.join(base_path, "AudioData",
                                    "features_30_sec.csv")
    return (audio_root, audio_task_type,
            audio_load_method, audio_labels_csv)


(image_root, image_task_type,
 image_load_method, image_labels_csv) = simulate_imagedata()
(audio_root, audio_task_type,
 audio_load_method, audio_labels_csv) = simulate_audiodata()

image_dataset = ImageDataset(root=image_root,
                             task_type=TaskType[image_task_type],
                             load_method=LoadMethod[image_load_method],
                             labels_csv=image_labels_csv)
audio_dataset = AudioDataset(root=audio_root,
                             task_type=TaskType[audio_task_type],
                             load_method=LoadMethod[audio_load_method],
                             labels_csv=audio_labels_csv)

print("=== Image Dataset ===")
if len(image_dataset) > 0:
    image, label = image_dataset[0]
    plt.imshow(image)
    plt.title("Sample Image with Label")
    plt.show()
    print(f"Image label: {label}")
    print(f"Task type: {image_task_type}")
    print(f"Image data loading method: {image_load_method}")
    train_size = input("What train size would you like to use for" +
                       " splitting the dataset? ")
    train_size = float(train_size)
    train, test = image_dataset.splitting(train_size=train_size)
    print(f"The dataset has been successfully split into {len(train)}" +
          f" training data and {len(test)} testing data.")
else:
    print("Image dataset is empty.")

print("\n=== Audio Dataset ===")
if len(audio_dataset) > 0:
    audio, label = audio_dataset[0]
    print(f"Audio label: {label}")
    print(f"Task type: {audio_task_type}")
    print(f"Audio data loading method: {audio_load_method}")
    train_size = input("What train size would you like to use for" +
                       " splitting the dataset? ")
    train_size = float(train_size)
    train, test = audio_dataset.splitting(train_size=train_size)
    print(f"The dataset has been successfully split into {len(train)}" +
          f" training data and {len(test)} testing data.")
else:
    print("Audio dataset is empty.")

print("\n=== BatchLoader ===")
image_batch_loader = BatchLoader(dataset=image_dataset, batch_size=10,
                                 random_batches=True, discard_last_batch=True)
audio_batch_loader = BatchLoader(dataset=audio_dataset, batch_size=10,
                                 random_batches=True, discard_last_batch=True)
image_batch_loader.create_batches()
audio_batch_loader.create_batches()

# Build preprocessing pipelines
image_pipeline = PreprocessingPipeline(CenterCrop(200, 200),
                                       RandomCrop(150, 150))
audio_pipeline = PreprocessingPipeline(RandomCropAudio(1),
                                       PitchShift(2))
# Apply preprocessing pipelines on Dataset samples
print("\n=== Applying Pipeline on BatchLoader Outputs for Images ===")
print("\nFive images were preprocessed using Center Crop" +
      " and Random Crop.")
for i in range(5):
    image_sample, _ = image_dataset[i]
    preprocessed_image = image_pipeline(image_sample)
    plt.imshow(preprocessed_image)
    plt.title('Preprocessed Image')
    plt.show()

print("\n=== Applying Pipeline on BatchLoader Outputs for Audio ===")
print("\nFive audio files were preprocessed with Random Cropping" +
      " and Pitch Shifting.")
for i in range(5):
    try:
        audio_sample, label = audio_dataset[i]
        audio_data, sample_rate = audio_sample

        if not np.issubdtype(audio_data.dtype, np.floating):
            maximum = np.max(np.abs(audio_data), axis=0)
            audio_data = audio_data.astype(np.float32) / maximum
        preprocessed_audio, sampling_rate = audio_pipeline((audio_data,
                                                            sample_rate))
        plt.figure(figsize=(12, 6))

        time = np.linspace(0, len(audio_data) / sample_rate,
                           num=len(audio_data))
        plt.subplot(2, 1, 1)
        plt.plot(time, audio_data)
        plt.title(f"Original Audio Waveform {label}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")

        time_preprocessed = np.linspace(0,
                                        len(preprocessed_audio) / sample_rate,
                                        num=len(preprocessed_audio))
        plt.subplot(2, 1, 2)
        plt.plot(time_preprocessed, preprocessed_audio)
        plt.title(f"Preprocessed Audio Waveform {label}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
