"""
filename: batchloader.py
Authors: Janan Jahed & Andrei Medesan
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))
import numpy as np
import logging
from dataorganisation import BaseDataset
from typing import Optional, Iterator, List, Any, Callable
from functools import wraps


class LoggingInfo:
    """
    Class for setting up a logger and logging information.
    Attributes:
    logger (logging.Logger): The logger object for logging information.
    Methods:
    _setup_logger(): Sets up the logger with a log file path, logging
    level, and format.
    """
    def __init__(self) -> None:
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        log_file_path = os.path.join(os.path.dirname(__file__), 'logInfo.log')
        logging.basicConfig(filename=log_file_path, level=logging.DEBUG,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
        return logging.getLogger(__name__)


class BatchLoader(LoggingInfo):
    """
    A class for creating batches from a given dataset.
    """
    def __init__(self, dataset: BaseDataset, batch_size: int,
                 random_batches: Optional[bool] = True,
                 discard_last_batch: Optional[bool] = True) -> None:
        super().__init__()
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
        if not isinstance(random_batches, bool):
            raise ValueError("random_batches must be a boolean value.")
        if not isinstance(discard_last_batch, bool):
            raise ValueError("discard_last_batch must be a boolean value.")
        self._dataset: BaseDataset = dataset
        self._batch_size: int = batch_size
        self._random_batches: bool = random_batches
        self._discard_last_batch: bool = discard_last_batch
        self._indices: np.ndarray = np.arange(len(dataset))
        if random_batches:
            np.random.shuffle(self._indices)

    def log_batch_info(func: Callable) -> Callable:
        """Decorator to log information about batch creation."""
        @wraps(func)
        def wrapper(self, *args: Any, **kwargs: Any) -> Iterator[List[Any]]:
            self.logger.info("Creating batches with batch size:" +
                             f" {self._batch_size}" +
                             f", Random batches: {self._random_batches}," +
                             "Discard last batch:" +
                             f" {self._discard_last_batch}")
            result = func(self, *args, **kwargs)
            print("Batches were created successfully." +
                  " Refer to the logInfo.log file for more information.")
            return result
        return wrapper

    @property
    def dataset(self) -> BaseDataset:
        """Getter for the dataset."""
        return self._dataset

    @property
    def batch_size(self) -> int:
        """Getter for the batch size."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        """Setter for the batch size."""
        if value <= 0:
            raise ValueError("Batch size must be greater than 0.")
        self._batch_size = value

    @property
    def random_batches(self) -> bool:
        """Getter for random_batches."""
        return self._random_batches

    @random_batches.setter
    def random_batches(self, value: bool) -> None:
        """Setter for random_batches."""
        self._random_batches = value

    @property
    def discard_last_batch(self) -> bool:
        """Getter for discard_last_batch."""
        return self._discard_last_batch

    @discard_last_batch.setter
    def discard_last_batch(self, value: bool) -> None:
        """Setter for discard_last_batch."""
        self._discard_last_batch = value

    @property
    def indices(self) -> np.ndarray:
        """Getter for indices."""
        return self._indices

    def __len__(self) -> int:
        """Returns the number of batches."""
        num_batches: int = len(self._dataset) // self._batch_size
        if not (self._discard_last_batch and
                len(self._dataset) % self._batch_size != 0):
            num_batches += 1
        return num_batches

    @log_batch_info
    def __iter__(self) -> Iterator[List[Any]]:
        """Iterator to iterate over batches."""
        current_batch: int = 0
        while current_batch < len(self):
            start_index: int = current_batch * self._batch_size
            end_index: int = start_index + self._batch_size

            if end_index > len(self._indices):
                if self._discard_last_batch:
                    break
                else:
                    end_index = len(self._indices)

            batch_indices: np.ndarray = self._indices[start_index:end_index]
            batch_data: List[Any] = [self._dataset[i] for i in batch_indices]

            current_batch += 1
            yield batch_data

    def create_batches(self) -> Iterator[List[Any]]:
        """Returns an iterator for batches."""
        return iter(self)
