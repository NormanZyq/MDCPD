import os
from abc import ABCMeta, abstractmethod


class DataLoader:
    __metaclass = ABCMeta

    @abstractmethod
    def load_data(self, data_path, **kwargs):
        raise NotImplementedError('Not implemented')
