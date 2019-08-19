import random as rd
import string
import os
import pandas as pd
from typing import List, Tuple
import PIL # keep it general, so as to be explicit when using the library

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from BalancedClusters import BalancedClusters
from FeatureSynthesis import FeatureSynthesis, FeatureTypes



class IncorrectTypeException(Exception):
    def __init__(self, variable_name, expected_type, actual_type):
        message = f'Expected type of {variable_name} to be {expected_type}' + f', but it was actually of type {actual_type}.'
        super(IncorrectTypeException, self).__init__(message)

class ImageTypes:
    RGB: str = 'RGB'


class Directions:
    """
    Simple, up down right left 2D directions.
    """
    Right = 'r'
    Left = 'l'
    Up = 'u'
    Down = 'd'


class _Direction:
    """
    Stores the horizontal and vertical components of a direction.
    """
    def __init__(self, quadrant):
        """
        parameters
            quadrant: the cartesian quadrant that the pixel
                      directed toward
        summary
            sets self.horizontal and self.vertical 
        """
        if quadrant not in (1, 2, 3, 4):
            raise Exception('Not a valid quadrant.')
        elif quadrant == 1:
            self.horizontal, self.vertical = Directions.Right, Directions.Up
        elif quadrant == 2:
            self.horizontal, self.vertical = Directions.Left, Directions.Up
        elif quadrant == 3:
            self.horizontal, self.vertical = Directions.Left, Directions.Down
        else:
            self.horizontal, self.vertical = Directions.Right, Directions.Down


class DirectionTypes:
    Horizontal = 'horizontal'
    Vertical = 'vertical'


class PixelLocation:
    def __init__(self):
        """ Basic constructor, no parameters"""
        self._pixel_directions = [] # will be a list of _Direction objects
        pass
    
    def copy(self):
        new_pixel_location = PixelLocation()
        new_pixel_location._pixel_directions = self._pixel_directions.copy()
        return new_pixel_location
    
    def add_pixel_direction(self, quadrant):
        """
        Constructs a _Direction object and appends it to the
        _pixel_directions list.
        """
        self._pixel_directions.append(_Direction(quadrant))
        
    def _get_num_pixels(self):
        """ Self explanatory. Returned variable is an integer."""
        return 4**len(self._pixel_directions)

    def _directory_to_binary(direction_type, directions):
        """
        parameters
            direction_type: specifies either horizontal or vertical
            directions: list of either horizontal or vertical directions
        returns
            binary string that represents the index for directions
            """
        valid_direction_types = (DirectionTypes.Horizontal, DirectionTypes.Vertical)
        if direction_type not in valid_direction_types:
            raise Exception('Invalid direction_type argument;'
                            + ' should have been one of the following:'
                            + f'\n{valid_direction_types}')
        
        if direction_type == DirectionTypes.Horizontal:
            direction_to_binary = {Directions.Left : 0, Directions.Right : 1}
            valid_directions = (Directions.Left, Directions.Right)
        else:
            direction_to_binary = {Directions.Up : 0, Directions.Down : 1}
            valid_directions = (Directions.Down, Directions.Up)
        
        binary_str = ''
        for direction in directions:
            if direction in valid_directions:
                binary_str += str(direction_to_binary[direction])
            else:
                raise Exception('Invalid value in directions. Expected direction to be in',
                               str(valid_directions) + f', but value was{direction}')
        
        return binary_str    
    
    def get_pixel_location(self):
        """
        Given the current pixel directions,
        find the location of the pixel in the
        image. Keep in mind, the number of
        specified directions indicates the size
        of the image.
        
        returns: tuple (i, j), where if the image were
                 a 2D array of pixels, the pixel would
                 be in the ith row and jth column. This
                 is standard array convention, NOT image
                 convention (column x row).
        """
        # First, get the vertical and horizontal directories
        horiz_dir = [direction.horizontal for direction in self._pixel_directions]
        vert_dir = [direction.vertical for direction in self._pixel_directions]
        # print(f'Here are the horizontal pixel directions: {horiz_dir}')
        # print(f'Here are the vertical pixel directions: {vert_dir}')
        
        # Now, get the binary string representing horizontal and vertical indices
        horiz_index_binary = PixelLocation._directory_to_binary(DirectionTypes.Horizontal, horiz_dir)
        vert_index_binary = PixelLocation._directory_to_binary(DirectionTypes.Vertical, vert_dir)
        
        # Now, convert binary to decimal
        binary = 2
        i, j = int(horiz_index_binary, binary), int(vert_index_binary, binary)
        return (i, j)



class FeatureDataTypes:
    Categorical = 'categorical'
    Numerical = 'numerical'
    Boolean = 'bool'
    Date = 'date'


class PreprocessingTools:

    @staticmethod
    def preprocess_data(df_inp: pd.DataFrame, cat_col_names: List[str]) -> pd.DataFrame:
        """
        :param df_inp:
        :param cat_col_names:
        :return:
        """
        # There are a lot of problems that come with one-hot encoding, so it will be
        # avoided for this implementation. One problem is that when we one-hot encode,
        # it increases the coding complexity for the transform function, becuase not
        # only do we need the feature name and the FeatureSynthesis class, but we also
        # need to know that the column was somehow one-hot encoded. Another problem is
        # the curse of dimensionality; if these features are one-hot encoded, then they
        # will generally be pretty sparse (especially if there are a lot of unique categories).
        # This will really mess up the cluster analysis, which is of course no good.
        df = PreprocessingTools.label_encode(df_inp, cat_col_names)
        df = PreprocessingTools.convert_from_bool_to_int(df)
        df = PreprocessingTools.scale_features(df, 0, 1)
        return df

    @staticmethod
    def convert_from_bool_to_int(df_inp: pd.DataFrame) -> pd.DataFrame:
        # Note: If we did "1 if x == True else 0", then all non-true df values would be "False", including non-bools
        df = df_inp.applymap(lambda x: 1 if x == True else x)
        df = df_inp.applymap(lambda x: 0 if x == False else x)
        return df

    @staticmethod
    def label_encode(df_inp: pd.DataFrame, cat_col_names: List[str]) -> pd.DataFrame:
        '''
        summary
            Replaces category columns with enumerations of the categories,
            i.e. assigns a number to each category. This is generally not
            thought of as a great practice, because it give categorical columns
            numeric appearance, and thus can be used in non-helpful ways by
            learning algorithms (like adding two categories together).
        parameters
            df_inp: pandas.DataFrame of features
            cat_col_names: list of strings, each string being the name of a categorical column
        returns
            pandas.DataFrame with category columns enumerated'''
        df = df_inp.copy()
        for cat_col in cat_col_names:
            df[cat_col] = LabelEncoder().fit_transform(df[cat_col])
        return df
    
    @staticmethod
    def one_hot_encode(df_inp: pd.DataFrame, cat_col_names: List[str]) -> pd.DataFrame:
        '''
        summary
            One-hot encodes for the categorical columns of a DataFrame
        parameters
            df_inp: pandas.DataFrame
            cat_col_names: list of strings, representing names of categorical columns
        returns
            a new pandas.DataFrame object
        '''
        df = df_inp.drop(columns=cat_col_names)
        for cat_col_name in cat_col_names:
            dummies_df = pd.get_dummies(df_inp[cat_col_name])
            dummies_cols = dummies_df.columns
            new_dummies_cols_names = [f'{cat_col_name}|{category}' for category in dummies_cols]
            dummies_df.columns = new_dummies_cols_names
            df = df.join(dummies_df)
        return df

    @staticmethod
    def scale_features(df_inp: pd.DataFrame, lower_bound: int, upper_bound: int) -> pd.DataFrame:
        """

        :param df_inp:
        :return:
        """
        scaler = MinMaxScaler((lower_bound, upper_bound))
        scaled_values = scaler.fit_transform(df_inp)

        return pd.DataFrame(scaled_values, index=df_inp.index, columns=df_inp.columns)


class ClusterAlgos:
    K_means = 'k-means'
    Agglomerative = 'agglomerative'


class ClusterTools:

    @staticmethod
    def __get_unique_column_name_helper(column_names, _num_times_repeated=0) -> str:
        '''
        Given a collection of column names, generate a column name that is
        not currently in the set of column names.
        '''
        
        if _num_times_repeated > 5:
            raise Exception("Something's not right; get_unique_column_name_helper repeated over 5 times. "
                           + f"There is an extremely tiny chance of this happening.")
            return
        else:
            # the more characters, the less likely the method will need to repeat
            num_characters = 30
            rand_str = ClusterTools.__random_string_helper(num_characters)
            if rand_str in column_names.values.tolist():
                return ClusterTools.__get_unique_column_name_helper(column_names, _num_times_repeated + 1)
            else:
                return rand_str

    @staticmethod
    def __random_string_helper(num_chars):
        valid_characters = string.ascii_letters + string.digits
        chars = [rd.choice(valid_characters) for _ in range(num_chars)]
        return ''.join(chars)

    @staticmethod
    def make_clusters(inp_data: pd.DataFrame, num_clusters, cluster_type):
        '''
        summary
            create clusters
        parameters
            inp_data: a pandas.DataFrame in typical ML format; PRESUMES CLEAN DATA
            num_clusters: integer number of clusters
            cluster_type: a flag-string clarifying which of the builtin clustering algorithms to use
            cat_col_names: a list of strings that are the names of the categorical columns
        returns
            a dictionary, mapping from cluster names to clusters; each cluster is just a pandas.DataFrame,
            with variable number of rows (since they're in clusters) and the same number of columns as the
            input data
        '''
        
        data: pd.DataFrame = inp_data.copy()
        label_col_name = ClusterTools.__get_unique_column_name_helper(data.columns)
        if cluster_type == ClusterAlgos.K_means:
            fit_kmeans = KMeans(num_clusters).fit(data)
            data[label_col_name] = fit_kmeans.labels_
        elif cluster_type == ClusterAlgos.Agglomerative:
            fit_agglomerative: AgglomerativeClustering = AgglomerativeClustering(num_clusters).fit(data)
            data[label_col_name] = fit_agglomerative.labels_
        else:
            raise Exception(f'cluster_type "{cluster_type}" is not a valid type. Try "{ClusterAlgos.K_means}"')
        
        # Create a column in the dataframe corresponding to the cluster label of the row,
        # then create clusters by grouping the dataframe by that column
        cluster_name_to_indices = data.groupby(label_col_name).groups  # a dictionary from label to row-indices
        clusters = dict()
        for cluster_name in cluster_name_to_indices.keys():
            indices = cluster_name_to_indices[cluster_name]
            clusters[cluster_name] = data.loc[indices].drop(columns=[label_col_name])
        return clusters


class ImageFromTable:
    """
    Objects that perform all actions to convert from tabular data to image data.
    """

    __DEFAULT_CLUSTER_TYPE = ClusterAlgos.K_means
    __MAX_PIXELS_MAGIC_VALUE = -1
    
    def __init__(self, feature_types: FeatureTypes, image_type: str, image_side_len: int = 16):
        """
        Set object properties
        :param feature_types:
        :param image_side_len:
        :param image_type:
        :param num_channels:
        :param channel_value_range: includes lower bound, excludes upper bound;
                this is the set [lower_bound, upper_bound)
        """

        # Initialize known values
        if image_type == ImageTypes.RGB:
            self.__num_channels: int = 3
            self.__channel_value_range: Tuple[float, float] = (0.0, 256.0)
        else:
            raise Exception('The parameter "image_type" was not valid. Check spelling')

        self.__image_type: str = image_type
        self.__image_side_len: int = image_side_len
        self.__total_num_features: int = (image_side_len ** 2) * self.__num_channels
        self._feature_types: FeatureTypes = feature_types
        self.__has_been_fit = False

        # Declare unknown values
        self.__feature_synthesizer: FeatureSynthesis = None
        self.__flattened_feature_name_image: List[str] = None
        return

    @staticmethod
    def __valid_image_side_len(image_side_len, max_pixels=__MAX_PIXELS_MAGIC_VALUE) -> bool:
        ''' 
        In order to be valid, the total number of pixels must
        be a power of four. This is because the feature placement
        algorithm divides the image into quadrants, subquadrants,
        subsubquadrants, ... to construct an image from the features.
        
        Optional max_pixels argument specifies maximum number of pixels;
        this defaults to 4096, the number of pixels in a 64x64 pixel image.
        '''
        num_pixels = image_side_len ** 2
        if (num_pixels > max_pixels) or (max_pixels == ImageFromTable.__MAX_PIXELS_MAGIC_VALUE):
            return False
        
        i = 0
        while 4 ** i <= num_pixels:
            if 4**i == num_pixels:
                return True
            else:
                i += 1
        return False

    @staticmethod
    def __make_blank_image_array(content_type, image_side_len, num_channels, max_feature_name_len):
        if content_type == str:
            return np.empty([image_side_len, image_side_len, num_channels], dtype=f'S{max_feature_name_len}')
        elif not isinstance(content_type, type(type(''))):
            raise IncorrectTypeException('content_type', type(type('')), type(content_type))
        else:
            raise Exception('Parameter \"content_type\" was not recognized.')
            
    def __populate_pixel_with_feature_names(self, location, x_transpose_df) -> None:
        ''' 
        parameters
            location: a PixelLocation object, describing where the pixel is located
            x_transpose_df: a pandas.DataFrame, with features as examples, and examples
                            as columns
        return
            None
        summary
            Puts feature names into their proper place in the image. However,
            there is not any order to it (i.e. feature names can be placed
            in any order, as opposed to going in alphabetical order, level, etc.).
        '''
        pixel_i, pixel_j = location.get_pixel_location()
        for k, feature_name in enumerate(x_transpose_df.index.values):
            try:
                self.__feature_name_image[pixel_i][pixel_j][k] = feature_name
            except Exception:
                print(f'Tried to populate pixel array in coordinate i: {pixel_i}, j: {pixel_j}.')
        return 

    def __num_examples(df_inp: pd.DataFrame):
        ''' 
        Number of examples is the number of rows in a regular
        array, and thus the number of columns in a transposed array.
        parameters
        :param: df:
        :returns: integer denoting the number of examples
        '''
        actual_type, df_type = type(df_inp), type(pd.DataFrame())
        
        if actual_type == df_type:
            return df_inp.shape[0]
        else:
            raise Exception(f'Expected transposed_arr to be of type {df_type}, but '
                           + f'the value passed was of type {actual_type}')

    def __populate_image_with_feature_names_helper(self, x_transpose_df, cluster_type, location_descriptor) -> None:
        """
        Groups feature names into an image by location; does so in-place to the instance variable
        self._feature_name_image.
        :param x_transpose_df:
        :param cluster_type:
        :param location_descriptor:
        :return:
        """
        if ImageFromTable.__num_examples(x_transpose_df) == self.__num_channels:
            # populate the three channels at the specified location in the picture
            self.__populate_pixel_with_feature_names(location_descriptor, x_transpose_df)
        else:
            # get balanced clusters for each quadrant
            # must be a dictionary from name to 2D numpy array
            clusters = ClusterTools.make_clusters(inp_data=x_transpose_df, num_clusters=4, cluster_type=cluster_type)
            num_clusters = len(clusters.keys())
            if num_clusters != 4:
                raise Exception(f"The number of clusters was incorrect! Expected {4} and got {num_clusters}."
                                "Try checking for similarity when generating features.")

            cluster_balancer = BalancedClusters(clusters, 'optimal')
            balanced_clusters = cluster_balancer.balance_clusters(verbose='none')

            # call recursively on each quadrant
            for i, key in enumerate(balanced_clusters.keys()):
                new_quadrant = i + 1  # quadrants begin indexing at 1; assigned arbitrarily
                new_location_descriptor = location_descriptor.copy()
                new_location_descriptor.add_pixel_direction(new_quadrant)
                self.__populate_image_with_feature_names_helper(balanced_clusters[key], cluster_type,
                                                                new_location_descriptor)
        return

    def __populate_image_with_feature_names(self, x_transpose_df, cluster_type):
        """
        Groups feature names into an image by location; does so in-place to the instance variable
        self._feature_name_image.
        :param x_transpose_df:
        :param cluster_type:
        :return:
        """
        self.__populate_image_with_feature_names_helper(x_transpose_df, cluster_type, PixelLocation())
        return
    
    def __make_flattened_feature_name_image(self) -> None:
        """

        :return:
        """
        column_names_bytes: List[np.bytes] = self.__feature_name_image.flatten('C')
        column_names_strs = [col_name_bytes.decode('utf-8') for col_name_bytes in column_names_bytes]
        self.__flattened_feature_name_image = column_names_strs
        return
    
    def fit(self, features, cluster_type=__DEFAULT_CLUSTER_TYPE):
        """
        summary
            Takes in features (X), NOT LABELS, and fits
            the class image to them. This method could
            take a very long time, if the image is large.
        parameters
            features: a DataFrame in typical data science format 
            (rows are examples, columns are features)
        returns
            nothing
        """
        self.__has_been_fit = False
        # Generate a bunch of features
        fs = FeatureSynthesis()
        synthetic_features = fs.fit(features, self._feature_types, self.__total_num_features,
                                    check_feature_similarity=True)
        
        cat_col_names = fs.get_feature_names(FeatureDataTypes.Categorical)
        synthetic_features = PreprocessingTools.preprocess_data(synthetic_features, cat_col_names)
        # synthetic_features has rows=examples, columns=features; transpose for __populate_image_with_feature_names
        synthetic_features_transpose = synthetic_features.transpose()

        # Initialize feature images
        max_feature_name_len = max( [len(feature_name) for feature_name in synthetic_features.columns.values] )
        self.__feature_name_image = ImageFromTable.__make_blank_image_array(
            str, self.__image_side_len, self.__num_channels, max_feature_name_len)
        self.__flattened_feature_name_image = self.__feature_name_image.flatten('C')

        # Do feature name thing to get image of feature names
        self.__populate_image_with_feature_names(synthetic_features_transpose,
                                                 cluster_type=cluster_type)
        
        # Flatten feature name image to be 1D
        self.__make_flattened_feature_name_image()

        # Save feature synthesizer
        self.__feature_synthesizer = fs
        return

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """

        :param features:
        :return:
        """

        # Make all of the synthetic features for this dataframe
        transformed_features: pd.DataFrame = self.__feature_synthesizer.transform(features)

        # Only use columns relevant for the image to be created; if all is correct (and "features" didn't have
        # excess/unused columns), then this should not remove any columns (i.e. should not change amount of info).
        # This operation also implicitly re-orders the columns, so pertinent_features will have the same column order
        # as the flattened_feature_name_image, making each row an image.
        pertinent_features = transformed_features[self.__flattened_feature_name_image]

        # transforms to same number of columns, but in range [0, 1] for all values
        cat_col_names = self.__feature_synthesizer.get_feature_names(FeatureDataTypes.Categorical)
        preprocessed_features = PreprocessingTools.preprocess_data(pertinent_features, cat_col_names)

        def scale(x):
            lower_bound, upper_bound = self.__channel_value_range
            value_range = upper_bound-lower_bound
            scaled = lower_bound + x*value_range
            if scaled > upper_bound:
                return upper_bound
            if scaled < lower_bound:
                return lower_bound
            else:
                return scaled

        scaled_features = preprocessed_features.applymap(scale)

        return scaled_features

    def __make_image(self, image_data: pd.Series) -> PIL.Image:
        """
        Creates a single image
        :param image_data:
        :return:
        """
        # Reshape into height-by-width-by-num_channels 3D array
        example_arr: np.array = image_data.values
        image_arr = np.reshape(example_arr, (self.__image_side_len, self.__image_side_len, self.__num_channels))

        # Convert into image
        img = PIL.Image.fromarray(image_arr, self.__image_type)
        return img

    def show_image(self, image_data: pd.Series) -> None:
        """
        Shows a single image
        :param image_data:
        :return:
        """
        img = self.__make_image(image_data)
        img.show()
        return

    def save_image(self, image_data: pd.Series, path: str) -> None:
        """
        Saves a single image
        :param image_data:
        :param path:
        :return:
        """
        img: PIL.Image = self.__make_image(image_data)
        img.save(path)
        return

    def save_all_images(self, image_data: pd.DataFrame, folder_path: str, image_names: np.array) -> None:
        """
        Saves all images to a specified folder.
        :param image_data:
        :param folder_path:
        :param image_names:
        :return:
        """
        assert isinstance(image_data, pd.DataFrame), 'Parameter "image_data" must be a pandas Dataframe.'
        num_image_rows = image_data.shape[0]
        assert num_image_rows == image_names.size, f'There are {image_data.shape[0]} potential images, and ' \
            f'{image_names.size} image names were given.'

        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        else:
            print(f'Warning: {folder_path} already exists, so new images will be added in with old images.')

        # TODO: Speed up with cython for-loop, or even try to parallelize it
        for i in range(num_image_rows):
            image_path = os.path.join(folder_path, image_names[i])
            self.save_image(image_data.iloc[i], image_path)

        return


if __name__ == '__main__':
    def make_examples_helper(n_rows):
        categories_1 = ('a', 'b', 'c', 'd', 'e')
        categories_2 = ('InstaMed', 'is', 'a', 'cool', 'company', 'check', 'it', 'out', 'sometime')
        rand_cats_1 = [rd.choice(categories_1) for _ in range(n_rows)]
        rand_cats_2 = [rd.choice(categories_2) for _ in range(n_rows)]

        df = pd.DataFrame({
            'example_numerical_col_1': np.random.rand(n_rows) * 50,
            'example_numerical_col_2': np.random.rand(n_rows) * 20,
            'example_categorical_col_1': rand_cats_1,
            'example_categorical_col_2': rand_cats_2,
            'example_boolean_col_1': np.random.randint(low=0, high=2, size=n_rows),
            'example_boolean_col_2': np.random.randint(low=0, high=2, size=n_rows)
        })
        feature_names = FeatureTypes(
            categorical=['example_categorical_col_1', 'example_categorical_col_2'],
            numerical=['example_numerical_col_1', 'example_numerical_col_2'],
            boolean=['example_boolean_col_1', 'example_boolean_col_2']
        )
        return df, feature_names

    # Get data
    example_df, example_feature_names = make_examples_helper(100)

    # Fit an image to the data
    img_side_len = 4
    img_type = ImageTypes.RGB
    image16by16by3 = ImageFromTable(example_feature_names, image_side_len=img_side_len, image_type=img_type)
    image16by16by3.fit(example_df, ClusterAlgos.Agglomerative)

    # Transform the image on new data
    new_example_features, _ = make_examples_helper(10)
    transformed_images = image16by16by3.transform(new_example_features)

    # See some of the resulting images
    max_num_images_shown = 5
    for i in range(min(transformed_images.shape[0], max_num_images_shown)):
        image_series = transformed_images.iloc[i] # just use the first element as an example
        image16by16by3.show_image(image_series)

    pass
