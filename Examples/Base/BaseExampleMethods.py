from ImageCreation import ImageFromTable, ClusterAlgos, ImageFolderOptions
from FeatureSynthesis import FeatureTypes
import pandas as pd
import numpy as np
from typing import List, Dict

class BaseExampleMethods:



    # @staticmethod
    # def create_images(image_transformation_parameters: Dict, folder_path: str, transform_image_names: np.array):
    #     BaseExampleMethods.get_image_transformations(**image_transformation_parameters)
    #     return

    @staticmethod
    def evaluate_classifier(predictions: np.array, labels: np.array) -> None:
        """

        :param predictions:
        :param labels:
        :return: Nothing, but creates console output.
        """
        accuracy = (predictions == labels).sum()
        print(f'Prediction accuracy was {accuracy*100}%.')
        return