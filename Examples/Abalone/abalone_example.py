import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
from FeatureSynthesis import FeatureTypes
from ImageCreation import ImageTypes, ClusterAlgos, ImageFolderOptions, ImageFromTable
import os

from Examples.Base.BaseExampleMethods import BaseExampleMethods


def main():
    # Get all data
    data_path = 'Data/abalone.data'
    column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight',
                    'Shell weight', 'Rings']
    all_data: pd.DataFrame = pd.read_csv(data_path, names=column_names)

    # Set the types of each feature
    label_column_name = 'Rings'
    feat_types = FeatureTypes(
        categorical=['Sex'],
        numerical=['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']
    )

    # Create images
    ImageFromTable.create_images(all_data=all_data, label_col_name=label_column_name, feature_types=feat_types)

    # TODO: Classify images

    # TODO: Evaluate performance

    return


if __name__ == '__main__':
    main()
