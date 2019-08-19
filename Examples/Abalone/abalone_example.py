import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from FeatureSynthesis import FeatureTypes
from ImageCreation import ImageTypes, ClusterAlgos, ImageFolderOptions
import os

from Examples.Base.BaseExampleMethods import BaseExampleMethods


def main():
    # Get all data (specific)
    data_path = 'Data/abalone.data'
    column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight',
                    'Shell weight', 'Rings']
    all_data: pd.DataFrame = pd.read_csv(data_path, names=column_names)

    # Split into fit and transform data; the two must be entirely separate, in order to avoid an info leak
    label_column_name = 'Rings'
    all_features = all_data.drop(columns=[label_column_name])
    all_labels = all_data[label_column_name]
    # Fit on 10% of the data, and then allow the other 90% to be transformed into images
    fit_features, transform_features, fit_labels, transform_labels = train_test_split(all_features, all_labels,
                                                                                      test_size=.9)

    # Set the types of each feature
    feat_types = FeatureTypes(
        categorical=['Sex'],
        numerical=['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']
    )

    # Set variables
    num_transformed_images = transform_features.shape[0]
    transform_img_names = np.array([f'index={i}|category={transform_labels.iloc[i]}.png'
                                    for i in range(num_transformed_images)])

    # Create images
    image_creation_parameters = {
        "fit_data_features": fit_features,
        "transform_data_features": transform_features,
        "feature_types": feat_types,
        "image_side_length": 4,
        "image_type": ImageTypes.RGB,
        "transform_image_folder_path": os.path.join(os.getcwd(), 'Images'),
        "transform_image_names": transform_img_names,
        "cluster_type": ClusterAlgos.Agglomerative,
        "image_folder_option": ImageFolderOptions.Replace_old_folder,
        "verbose": True,
    }

    BaseExampleMethods.create_images(**image_creation_parameters)
    # TODO: Classify images

    # TODO: Evaluate performance

    return


if __name__ == '__main__':
    main()
