"""
file: preprocessing.py
Authors: Janan Jahed & Andrei Medesan
"""
import random
from PIL import Image
import librosa
import numpy as np
from abc import ABC, abstractmethod


class PreprocessingTechniqueABC(ABC):
    """
    Abstract base class for all preprocessing techniques.
    Defines the generic behavior for preprocessing operations.
    """

    @abstractmethod
    def __call__(self, data: any) -> any:
        """
        Apply the preprocessing technique to the input data.
        Parameters:
        - data: The data to be preprocessed.
        Returns:
        - The preprocessed data.
        """
        pass


class CenterCrop(PreprocessingTechniqueABC):
    """
    A preprocessing technique for images that crops an image to a specified
    width and height, centered on the middle of the image.
    """
    def __init__(self, width: int, height: int):
        """
        Initializes the CenterCrop preprocessing technique with the desired
        width and height.
        Parameters:
        - width: Desired width of the cropped image.
        - height: Desired height of the cropped image.
        """
        self.width = width
        self.height = height

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Applies the center crop to the given image.
        Parameters:
        - image: The image to be cropped.
        Returns:
        - The cropped image.
        """
        original_width, original_height = image.size
        left = (original_width - self.width) // 2
        top = (original_height - self.height) // 2
        right = (original_width + self.width) // 2
        bottom = (original_height + self.height) // 2
        return image.crop((left, top, right, bottom))


class RandomCrop(PreprocessingTechniqueABC):
    """
    A preprocessing technique for images that crops a random portion of the
    image to a specified width and height.
    """
    def __init__(self, width: int, height: int):
        """
        Initializes the RandomCrop preprocessing technique with the desired
        width and height.
        Parameters:
        - width: Desired width of the cropped image.
        - height: Desired height of the cropped image.
        """
        self.width = width
        self.height = height

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Applies a random crop to the given image.
        Parameters:
        - image: The image to be cropped.
        Returns:
        - The cropped image.
        """
        original_width, original_height = image.size
        left = random.randint(0, max(0, original_width - self.width))
        top = random.randint(0, max(0, original_height - self.height))
        right = min(left + self.width, original_width)
        bottom = min(top + self.height, original_height)
        return image.crop((left, top, right, bottom))


class RandomCropAudio(PreprocessingTechniqueABC):
    """
    A preprocessing technique for audio data that crops a random portion
    of the audio clip to a specified duration.
    """
    def __init__(self, duration: float):
        """
        Initializes the RandomCropAudio preprocessing technique with the
        desired duration.
        Parameters:
        - duration: Desired duration of the cropped audio in seconds.
        """
        self.duration = duration

    def __call__(self, data: tuple[np.ndarray, int]) -> tuple[np.ndarray, int]:
        """
        Applies a random crop to the given audio data.
        Parameters:
        - data: A tuple of (audio_data, sampling_rate).
        Returns:
        - A tuple of the cropped audio data and its sampling rate.
        """
        audio_data, sampling_rate = data
        if len(audio_data) <= self.duration * sampling_rate:
            return audio_data, sampling_rate
        start = random.randint(
            0, len(audio_data) - int(self.duration * sampling_rate))
        end = start + int(self.duration * sampling_rate)
        return audio_data[start:end], sampling_rate


class PitchShift(PreprocessingTechniqueABC):
    """
    A preprocessing technique for audio data that shifts the pitch of the
    audio clip by a specified factor.
    """
    def __init__(self, pitch_factor: float):
        """
        Initializes the PitchShift preprocessing technique with the desired
        pitch factor.
        Parameters:
        - pitch_factor: Factor by which the pitch should be shifted. Positive
        values shift the pitch up, negative values shift it down.
        """
        self.pitch_factor = pitch_factor

    def __call__(self, data: tuple[np.ndarray, int]) -> tuple[np.ndarray, int]:
        """
        Applies a pitch shift to the given audio data.
        Parameters:
        - data: A tuple of (audio_data, sampling_rate).
        Returns:
        - A tuple of the pitch-shifted audio data and its sampling rate.
        """
        audio_data, sampling_rate = data
        shifted_audio = librosa.effects.pitch_shift(
            audio_data, sr=sampling_rate, n_steps=self.pitch_factor)
        return shifted_audio, sampling_rate


class PreprocessingPipeline:
    """
    A pipeline that allows for the sequential application of multiple
    preprocessing techniques to data.
    """
    def __init__(self, *steps: PreprocessingTechniqueABC):
        """
        Initializes the PreprocessingPipeline with a sequence of preprocessing
        steps.
        Parameters:
        - steps: A variable number of preprocessing techniques that will be
        applied in sequence.
        """
        self.steps = steps

    def __call__(self, data: any) -> any:
        """
        Applies each preprocessing step to the data in sequence.
        Parameters:
        - data: The data to be preprocessed.
        Returns:
        - The preprocessed data.
        """
        for step in self.steps:
            data = step(data)
        return data
