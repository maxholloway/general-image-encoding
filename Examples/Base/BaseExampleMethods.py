from ImageCreation import ImageFromTable, ClusterAlgos, ImageFolderOptions
from FeatureSynthesis import FeatureTypes
import pandas as pd
import numpy as np
from typing import List, Dict

class BaseExampleMethods:

    @staticmethod
    def create_images(fit_data_features: pd.DataFrame, transform_data_features: pd.DataFrame,
                      feature_types: FeatureTypes, image_side_length: int, image_type: str,
                      transform_image_folder_path: str, transform_image_names: np.array,
                      image_folder_option: str = ImageFolderOptions.Create_new_folder,
                      cluster_type: str = ClusterAlgos.Agglomerative, verbose: bool = False) -> None:
        """
        Creates images from tabular data, and stores them in a folder as ".png" files.
        :param fit_data_features:
        :param transform_data_features:
        :param feature_types:
        :param image_side_length:
        :param image_type:
        :param transform_image_folder_path:
        :param transform_image_names:
        :param image_folder_option:
        :param cluster_type:
        :param verbose:
        :return:
        """

        assert (fit_data_features.shape[0] >= 1) and (transform_data_features.shape[0] >= 1), \
            "There were not enough examples given."

        # Create ImageFromTable object
        img_from_table = ImageFromTable(feature_types=feature_types, image_side_len=image_side_length,
                                        image_type=image_type)

        # Fit the object; THE DATA USED FOR THE MODEL SHOULD BE SEPARATED FROM THAT BEING USED IN ALGORITHMS, AKA
        # IT IS NECESSARY TO USED SEPARATE fit_data_features AND transform_data_features.
        img_from_table.fit(fit_data_features, cluster_type)

        # Transform the object
        images: pd.DataFrame = img_from_table.transform(transform_data_features)

        # Show the first image that was created
        if verbose:
            first_image: pd.Series = images.iloc[0]
            img_from_table.show_image(first_image)

        # Save all of the images that were created
        img_from_table.save_all_images(images, transform_image_folder_path, transform_image_names,
                                       image_folder_option=image_folder_option)

        return

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