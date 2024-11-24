# Report

This repository hosts a Python application centered on data processing and preprocessing, utilizing the principles of object-oriented programming (OOP). It is designed to facilitate the handling of datasets, log activities, and apply preprocessing methods to both image and audio data for regression or calssification tasks.

The project's structure is modular, with each module dedicated to a particular aspect of the data processing workflow. Data loading and manipulation is handled in the _dataorganisation.py_ file, the creation of batches for the training data is handled in _batchloader.py_ and the preprocessing techniques applied to both image and audio data are handled in the _preprocessing.py_ file.
The data for this project is sourced from Kaggle and comprises an audio folder with 10 genres, each containing 100 audio files of 30-second clips. [GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). This data was also added in the repository as a folder under the name of _AudioData_. The data utilized for images was also uploaded on this repository --_ImageData_-- which was sourced also from Kaggle [Color Polygon Images](https://www.kaggle.com/datasets/gonzalorecioc/color-polygon-images). The data comprises 10.000 simple image data files and a CSV file.

Key OOP principles used:

1. **Abstraction:** Abstract methods were introduced in the BaseDatset, PreprocessingTechiniqueABC and LoggingInfo classes to provide a blueprint for the other subclasses that inherit from them. These methods represent the core functions that ought to be implemented in the subsequent classes.

2. **Inheritance:** Several classes, such as ImageDataset, AudioDataset, BatchLoader, and the four preprocessing techniques implementations inherit functionality from the abstract ones. For instance, both ImageDataset and AudioDataset inherit from BaseDataset.

3. **Encapsulation:** Classes encapsulate related attributes and methods within a single unit. For example, the BaseDataset class uses underscore-prefixed attributes (like *_root*, *_data*, *_task_type*, *_load_method*, and *_labels_csv*) to denote that these attributes are intended for internal use within the class. It then provides public getter and setter methods for these attributes, which are implemented as properties in Python.

4. **Polymorphism** Polymorphism is displayed through the __getitem__, _load_data_eagerly and _load_data_lazily methods in the ImageDatset and AudioDataset classes, where they were implemented differently in each class, handling the audio and image data files.

**Program Structure**

1. *dataorganization.py*

This file handles the loading, splitting and certain manipulations of the data (__len__, __getitem__).

Classes:
- TaskType: Enum defining task types - classification or regression.
- LoadMethod: Enum defining loading methods - lazy or eager.
- BaseDataset: contains the abstract methods, the __init__ function for all the subclasses and the splitting method used for either type of data.
- ImageDataset: contains the implementations of the abstract methods from the BaseDataset that are adjusted to support image data files.
- AudioDataset: contains the implementations of the abstract methods inherited which are adjusted according to the audio data files.

2.*batchloader.py*

This file handles the creation of batches on the training data.

Classes:
- LoggingInfo: creation of a logger for describing log information.
- BatchLoader: Class for creating batches from a given dataset.

3. *preprocessing.py*

This files handles the four preprocessing techniques that can be applied to either images or audios. Notably, two techiniques are addressed for image data files, while the other two are assigned to audio data files.

Classes:
- PreprocessingTechniqueABC: Abstract base class for preprocessing techniques.
- CenterCrop: Class for performing center cropping on an image.
- RandomCrop: Class for performing random cropping on an image.
- RandomCropping: Class for performing random cropping on an audio signal.
- PitchShifting: Class for performing pitch shifting on an audio signal.
- PreprocessingPipeline: Class for performing the preprocessing techinique sequentially on the data files using a specified number of repetitions.

4. *main.py*

The main file demonstrates the usage of the other files by accessing the data folders in the repository and showcasing all the methods that can be applied for the data's specifications (image or audio).

**Code Workflow**

The program has specific functions integrated in the main that simulate the gathering of information from the data folders such as the root, task type, load method (which can be changed), and the CSV root file. These functions prepare the respective datasets for loading and showcase one sample with its label, visual/ audio represenation, task type and load method used. Further, the code requires the user's input for the desired train size in order to split the data accordingly. This value must be of type float between [0.1 and 1.0].
Furthermore, the batch loader is initiated on both datasets and further information such as the size of the batches can be found in the _LogInfo.log_ file. For the preprocessing phase, five data files from each dataset are selected in order to display the chosen preprocessing techiniques and the use of the pipepline method. Notably, for the audio dataset the user can decide whether to play or not the audio file.

**Run our Program**
1. Clone this repository to your machine.
2. Ensure you have the required libraries installed (NumPy, Pillow, Librosa, and simpleaudio).
3. Run the main.py script. Answer the question prompted by the code.
