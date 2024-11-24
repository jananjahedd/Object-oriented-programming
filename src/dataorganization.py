"""
filename: dataorganization.py
Authors: Janan Jahed & Andrei Medesan
"""

import os
import csv
import librosa
import warnings
from typing import Tuple, List, Optional, Union, TypeVar
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from enum import Enum
from abc import ABC, abstractmethod
import simpleaudio as sa

# restrictions for some of the librosa unwanted warnings
warnings.filterwarnings(
    "ignore",
    message="PySoundFile failed. Trying audioread instead.")
warnings.filterwarnings(
    "ignore",
    category=FutureWarning, message=".*__audioread_load.*")
Im = TypeVar('Im', Image, float)
Au = TypeVar('Au', np.ndarray, int, float)


class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class LoadMethod(Enum):
    EAGER = "eager"
    LAZY = "lazy"


class BaseDataset(ABC):
    """
    The class is an abstract base class that provides common
    functionality for loading and accessing data.
    Properties for `root`, `task_type`, `load_method`, and
    `labels_csv`, with corresponding getters and setters.
    """
    def __init__(self, root: str = "",
                 task_type: TaskType = None,
                 load_method: LoadMethod = None,
                 labels_csv: Optional[str] = None) -> None:
        self._data = []
        self._data_paths = []
        self._labels = []
        self._root = root
        self._task_type = task_type
        self._load_method = load_method
        self._labels_csv = labels_csv

        if self._load_method == LoadMethod.EAGER:
            self._data = self._load_data_eagerly()
        elif self._load_method == LoadMethod.LAZY:
            self._data_paths, self._labels = self._load_data_lazy()

    @property
    def root(self) -> str:
        return self._root

    # Setter for root
    @root.setter
    def root(self, value: str) -> None:
        self._root = value

    # Getter for task_type
    @property
    def task_type(self) -> TaskType:
        return self._task_type

    # Setter for task_type
    @task_type.setter
    def task_type(self, value: TaskType) -> None:
        self._task_type = value

    # Getter for load_method
    @property
    def load_method(self) -> LoadMethod:
        return self._load_method

    # Setter for load_method
    @load_method.setter
    def load_method(self, value: LoadMethod) -> None:
        self._load_method = value

    # Getter for labels_csv
    @property
    def labels_csv(self) -> Optional[str]:
        return self._labels_csv

    # Setter for labels_csv
    @labels_csv.setter
    def labels_csv(self, value: Optional[str]) -> None:
        self._labels_csv = value

    @abstractmethod
    def _load_data_lazy(self) -> Tuple[List[str],
                                       Optional[List[Union[float, str]]]]:
        pass

    @abstractmethod
    def _load_data_eagerly(self) -> List[Union[Im, Au]]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Union[Im, Au]:
        pass

    def splitting(self,
                  train_size: float) -> Tuple['BaseDataset', 'BaseDataset']:
        """
        The splitting function splits the data into two datasets.
        The train size is selected by the user and returns a tuple
        containing two BaseDatasets.
        """
        train_dataset = self.__class__(
            self._root, self._task_type, self._load_method, self._labels_csv)
        test_dataset = self.__class__(
            self._root, self._task_type, self._load_method, self._labels_csv)

        if self._load_method == LoadMethod.LAZY:
            X_train, X_test, y_train, y_test = train_test_split(
                self._data_paths, self._labels, train_size=train_size)
            train_dataset._data_paths, train_dataset._labels = X_train, y_train
            test_dataset._data_paths, test_dataset._labels = X_test, y_test
        elif self._load_method == LoadMethod.EAGER:
            data_train, data_test = train_test_split(
                self._data, train_size=train_size)
            train_dataset._data = data_train
            test_dataset._data = data_test

        return train_dataset, test_dataset


class ImageDataset(BaseDataset):
    """
    The class is a subclass of `BaseDataset`
    that specifically handles image datasets.
    It implements the abstract methods using PIL
    library to load images. It also provides
    an additional method for getting the length and
    item at a specific index of
    the dataset.
    """
    def _load_data_lazy(self) -> Tuple[List[str],
                                       Optional[List[Union[float, str]]]]:
        """The function loads the data files in a lazy fashion."""
        labels_dict = {}
        if self._labels_csv:
            with open(self._labels_csv, mode='r') as csvfile:
                csvreader = csv.DictReader(csvfile)
                for row in csvreader:
                    filename = row['filename']
                    label = row['sides']
                    if self._task_type == TaskType.REGRESSION:
                        label = float(label)
                    else:
                        int(label)
                    labels_dict[filename] = label

            data_paths = [
                os.path.join(self._root,
                             filename) for filename in os.listdir(self._root)]
            labels = [labels_dict.get(os.path.basename(path), None)
                      for path in data_paths
                      if os.path.basename(path) in labels_dict]

            return data_paths, labels

    def _load_data_eagerly(self) -> List[Im]:
        """The function handles the data file loading
        in an eager fashion"""
        data_paths, labels = self._load_data_lazy()
        loaded_data = [
            (Image.open(path).convert('RGB'), labels[idx])
            for idx, path in enumerate(data_paths)]
        return loaded_data

    def __len__(self) -> int:
        """The magic function overrides the len() built-in function"""
        if self._load_method == LoadMethod.LAZY:
            return len(self._data_paths)
        else:
            return len(self._data)

    def __getitem__(self, index: int) -> Tuple[Im, int | float]:
        """The function returns the item at the specified index."""
        if self._load_method == LoadMethod.LAZY:
            path = self._data_paths[index]
            image = Image.open(path).convert('RGB')
            label = self._labels[index]
            return image, label
        else:
            return self._data[index]


class AudioDataset(BaseDataset):
    """
    The class is another subclass of `BaseDataset`
    that handles audio datasets.
    It implements the abstract methods using librosa
    library to load audio files.
    It also provides additional methods for playing audio files
    and getting the length
    and item at a specific index of the dataset.
    """
    def _load_data_lazy(self) -> Tuple[List[str],
                                       Optional[List[Union[float, str]]]]:
        """Function that handles lazy loading fashion."""
        data_paths = []
        labels = []
        labels_dict = {}
        if self._labels_csv:
            with open(self._labels_csv, mode='r') as csvfile:
                csvreader = csv.reader(csvfile)
                next(csvreader, None)
                for row in csvreader:
                    if self._task_type == TaskType.REGRESSION:
                        labels_dict[row[0]] = float(row[1])
                    else:
                        labels_dict[row[0]] = row[1]

        for genre_dir in os.listdir(self._root):
            genre_path = os.path.join(self._root, genre_dir)
            if os.path.isdir(genre_path):
                for filename in os.listdir(genre_path):
                    if filename.endswith('.wav'):
                        full_path = os.path.join(genre_path, filename)
                        data_paths.append(full_path)
                        label = labels_dict.get(filename, None)
                        labels.append(label)
        return data_paths, labels

    @staticmethod
    def load_audio_file(path: str) -> Au:
        """Function to load the audio files using librosa."""
        try:
            data, samplerate = librosa.load(path, sr=None, mono=False)
            data = (data * 32767).astype(np.int16)
            return data, samplerate
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None, None

    @staticmethod
    def play_audio_file(data: List[Au], samplerate: float,
                        filename: str) -> None:
        """Function to display the audio file selected
        in the original format."""
        print(f"Playing audio file: {filename}")
        try:
            audio_bytes = data.tobytes()
            num_channels = 2 if data.ndim > 1 else 1
            play_obj = sa.play_buffer(audio_bytes,
                                      num_channels=num_channels,
                                      bytes_per_sample=2,
                                      sample_rate=samplerate)
            play_obj.wait_done()
        except Exception as e:
            print(f"Error playing audio: {e}")

    def _load_data_eagerly(self) -> List[Au]:
        """Function that handles the eager data loading for audio files."""
        eager_data = []
        labels_dict = {}
        if self._labels_csv:
            with open(self._labels_csv, mode='r') as csvfile:
                csvreader = csv.reader(csvfile)
                next(csvreader, None)
                for row in csvreader:
                    if self._task_type == TaskType.REGRESSION:
                        labels_dict[row[0]] = float(row[1])
                    else:
                        labels_dict[row[0]] = row[1]

        for genre_dir in os.listdir(self._root):
            genre_path = os.path.join(self._root, genre_dir)
            if os.path.isdir(genre_path):
                for filename in os.listdir(genre_path):
                    if filename.endswith('.wav'):
                        full_path = os.path.join(genre_path, filename)
                        result = AudioDataset.load_audio_file(full_path)
                        audio, sample_rate = result
                        if audio is not None:
                            label = labels_dict.get(filename, None)
                            entry = ((audio, sample_rate), label, filename)
                            eager_data.append(entry)
        return eager_data

    def __len__(self) -> int:
        """Function that overrides the built-in len() function."""
        if self._load_method == LoadMethod.LAZY:
            return len(self._data_paths)
        else:
            return len(self._data)

    def __getitem__(self, index: int) -> Tuple[Au, int | float]:
        """Function that returns the data file at the specified
        index and also display the file through the user's input."""
        if self._load_method == LoadMethod.LAZY:
            path = self._data_paths[index]
            filename = os.path.basename(path)
            audio, sample_rate = self.load_audio_file(path)
            label = self._labels[index]
        else:
            audio, sample_rate = self._data[index][0]
            label = self._data[index][1]
            filename = self._data[index][2]

        user_input = input("Do you want to play the audio file? (y/n) ")
        if user_input == "y":
            AudioDataset.play_audio_file(audio, sample_rate, filename)

        return (audio, sample_rate), label
