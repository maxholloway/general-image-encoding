class IncorrectTypeException(Exception):
    def __init__(self, variable_name, expected_type, actual_type):
        message = f'Expected type of {variable_name} to be {expected_type}' + f', but it was actually of type {actual_type}.'
        super(IncorrectTypeException, self).__init__(message)

class Directions:
    '''
    Simple, up down right left 2D directions.
    '''
    Right = 'r'
    Left = 'l'
    Up = 'u'
    Down = 'd'

class _Direction:
    '''
    Stores the horizontal and vertical components of a direction.
    '''
    def __init__(self, quadrant):
        '''
        parameters
            quadrant: the cartesian quadrant that the pixel
                      directed toward
        summary
            sets self.horizontal and self.vertical 
        '''
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
        ''' Basic constructor, no parameters'''
        self._pixel_directions = [] # will be a list of _Direction objects
        pass
    
    def copy(self):
        new_pixel_location = PixelLocation()
        new_pixel_location._pixel_directions = self._pixel_directions.copy()
        return new_pixel_location
    
    def add_pixel_direction(self, quadrant):
        ''' 
        Constructs a _Direction object and appends it to the
        _pixel_directions list.
        '''
        self._pixel_directions.append(_Direction(quadrant))
        
    def _get_num_pixels(self):
        ''' Self explanatory. Returned variable is an integer.'''
        return 4**len(self._pixel_directions)
    
    
    def _directory_to_binary(direction_type, directions):
        '''
        parameters
            direction_type: specifies either horizontal or vertical
            directions: list of either horizontal or vertical directions
        returns
            binary string that represents the index for directions
            '''
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
        '''
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
        '''
        # First, get the vertical and horizontal directories
        horiz_dir = [direction.horizontal for direction in self._pixel_directions]
        vert_dir = [direction.vertical for direction in self._pixel_directions]
        print(f'Here are the horizontal pixel directions: {horiz_dir}')
        print(f'Here are the vertical pixel directions: {vert_dir}')
        
        # Now, get the binary string representing horizontal and vertical indices
        horiz_index_binary = PixelLocation._directory_to_binary(DirectionTypes.Horizontal, horiz_dir)
        vert_index_binary = PixelLocation._directory_to_binary(DirectionTypes.Vertical, vert_dir)
        
        # Now, convert binary to decimal
        binary = 2
        i, j = int(horiz_index_binary, binary), int(vert_index_binary, binary)
        return (i, j)

import random as rd
import string
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from BalancedClusters import BalancedClusters
from FeatureSynthesis import FeatureSynthesis, FeatureTypes

class FeatureDataTypes:
    Categorical = 'categorical'
    Numerical = 'numerical'
    Boolean = 'bool'
    Date = 'date'

from sklearn.preprocessing import LabelEncoder
class PreprocessingTools:
    def preprocess_data(df_inp, cat_col_names):
        '''
        TODO: Add feature normalization 
        '''
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
        #fstr = f'\n\n\nData types are {df.dtypes}'
        #fstr = df['example_categorical_col_1']
        #print(fstr)
        return df
    def convert_from_bool_to_int(df_inp):
        # Note: If we did "1 if x == True else 0", then all non-true df values would be "False"
        df = df_inp.applymap(lambda x: 1 if x == True else x)
        df = df_inp.applymap(lambda x: 0 if x == False else x)
        return df
    
    def label_encode(df_inp, cat_col_names):
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
    
    def one_hot_encode(df_inp, cat_col_names):
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

class ClusterAlgos:
    K_means = 'k-means'

class ClusterTools:
    def _get_unique_column_name_helper(column_names, _num_times_repeated=0):
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
            rand_str = ClusterTools._random_string_helper(num_characters)
            if rand_str in column_names.values:
                return ClusterTools._get_unique_column_name_helper(column_names, _num_times_repeated+1)
            else:
                return rand_str
    
        
    def _random_string_helper(num_chars):
        valid_characters = string.ascii_letters + string.digits
        chars = [rd.choice(valid_characters) for _ in range(num_chars)]
        return ''.join(chars)
    
    def make_clusters(inp_data, num_clusters, cluster_type):
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
        
        data = inp_data.copy()
        label_col_name = ClusterTools._get_unique_column_name_helper(data.columns)
        if cluster_type == ClusterAlgos.K_means:
            fit_kmeans = KMeans(num_clusters).fit(data)
            data[label_col_name] = fit_kmeans.labels_
        else:
            raise Exception(f'cluster_type "{cluster_type}" is not a valid type. Try "{ClusterAlgos.K_means}"')
        
        # Create a column in the dataframe corresponding to the cluster label of the row,
        # then create clusters by grouping the dataframe by that column
        cluster_name_to_indices = data.groupby(label_col_name).groups # a dictionary from label to row-indices
        clusters = dict()
        for cluster_name in cluster_name_to_indices.keys():
            indices = cluster_name_to_indices[cluster_name]
            clusters[cluster_name] = data.loc[indices].drop(columns=[label_col_name])
        return clusters

class ImageFromTable:
    _DEFAULT_CLUSTER_TYPE = ClusterAlgos.K_means
    
    def __init__(self, feature_types, image_side_len=16, num_channels=3):
        self.image_side_len = image_side_len
        self.num_channels = num_channels
        self._total_num_features = (image_side_len**2)*num_channels
        self._feature_name_image = ImageFromTable._make_blank_image_array(
            type(" "), image_side_len, num_channels)
        self._flattened_feature_name_image = self._feature_name_image.flatten('C')
        self._feature_types = feature_types
        return
    
    def _valid_image_side_len(image_side_len, max_pixels=4096):
        ''' 
        In order to be valid, the total number of pixels must
        be a power of four. This is because the feature placement
        algorithm divides the image into quadrants, subquadrants,
        subsubquadrants, ... to construct an image from the features.
        
        Optional max_pixels argument specifies maximum number of pixels;
        this defaults to 4096, the number of pixels in a 64x64 pixel image.
        '''
        num_pixels = image_side_len ** 2
        if num_pixels > max_pixels: 
            return False
        
        i = 0
        while 4 ** i <= num_pixels:
            if 4**i == num_pixels:
                return True
            else:
                i += 1
        return False

    def _make_blank_image_array(content_type, image_side_len, num_channels):
        if content_type == type(str()):
            max_num_chars = 100
            return np.empty([image_side_len, image_side_len, 3], dtype=f'S{max_num_chars}')
        elif type(content_type) != type(type('')):
            raise IncorrectTypeException('content_type', type(type('')), type(content_type))
        else:
            raise Exception('Parameter \"content_type\" was not recognized.')
            
    def _populate_pixel_with_feature_names(self, location, x_transpose_df):
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
            self._feature_name_image[pixel_i][pixel_j][k] = feature_name
        return 
    
    
    def _num_examples(df_inp):
        ''' 
        Number of examples is the number of rows in a regular
        array, and thus the number of columns in a transposed array.
        parameters
            df: a dataframe
        returns
            integer denoting the number of examples
        '''
        actual_type, df_type = type(df_inp), type(pd.DataFrame())
        
        if actual_type == df_type:
            return df_inp.shape[0]
        else:
            raise Exception(f'Expected transposed_arr to be of type {df_type}, but '
                           + f'the value passed was of type {actual_type}')
    
    def _populate_image_with_feature_names(self, x_transpose_df, location_descriptor=PixelLocation(),
                                          cluster_type = _DEFAULT_CLUSTER_TYPE,
                                          num_tries=100):
        '''
        summary
            Groups feature names into an image by location; does so in-place to the instance variable
            self._feature_name_image
        parameters
            x_transpose_df: dataframe where rows are 
            location_description: a list of strings of 
        returns
            Nothing, just populates self._feature_name_image
        '''
        if ImageFromTable._num_examples(x_transpose_df) == self.num_channels:
            # populate the three channels at the specified location in the picture
            
            self._populate_pixel_with_feature_names(location_descriptor, x_transpose_df)
        else:
            # get balanced clusters for each quadrant
            # must be a dictionary from name to 2D numpy array
            clusters = ClusterTools.make_clusters(inp_data=x_transpose_df,
                                     num_clusters=4, cluster_type=cluster_type)
            num_times_tried_clustering = 1
            while len(clusters.keys()) != 4 and num_times_tried_clustering < num_tries:
                clusters = ClusterTools.make_clusters(inp_data=x_transpose_df,
                             num_clusters=4, cluster_type=cluster_type)
                num_times_tried_clustering += 1
            
            assert len(clusters.keys()) == 4, f'{len(clusters.keys())} cluster keys'
            
            max_iterations = 1000 # TODO: make function to get it depending on image size
            cluster_balancer = BalancedClusters(clusters,'optimal')
            balanced_clusters = cluster_balancer.balance_clusters(
                max_iterations=max_iterations, verbose='none')
            
            assert len(balanced_clusters.keys()) == 4, f"{len(balanced_clusters.keys())} balanced_cluster keys"
            # call recursively on each quadrant
            for i, key in enumerate(balanced_clusters.keys()): 
                new_quadrant = i+1 # quadrants begin indexing at 1; assigned arbitrarily
                new_location_descriptor = location_descriptor.copy()
                new_location_descriptor.add_pixel_direction(new_quadrant)
                self._populate_image_with_feature_names(x_transpose_df=balanced_clusters[key],
                                                  location_descriptor=new_location_descriptor)
        return
    
    def _make_flattened_feature_name_image(self):
        self._flattened_feature_name_image = self._feature_name_image.flatten('C')
    
    def fit(self, features, cluster_type=_DEFAULT_CLUSTER_TYPE):
        '''            
        summary
            Takes in features (X), NOT LABELS, and fits
            the class image to them. This method could
            take a very long time, if the image is large.
        parameters
            features: a DataFrame in typical data science format 
            (rows are examples, columns are features)
        returns
            nothing
        '''
        # Generate a bunch of features
        fs = FeatureSynthesis(feature_names=self._feature_types, 
                              total_num_features=self._total_num_features)
        synthetic_features = fs.synthesize_features(features)
        
        cat_col_names = fs.get_col_names(FeatureDataTypes.Categorical)
        synthetic_features = PreprocessingTools.preprocess_data(synthetic_features, cat_col_names)
        # synthetic_features has rows=examples, columns=features; transpose for _populate_image_with_feature_names
        synthetic_features_transpose = synthetic_features.transpose()
        # Do feature name thing to get image of feature names
        self._populate_image_with_feature_names(synthetic_features_transpose,
                                                cluster_type=cluster_type)
        
        # Flatten feature name image to be 1D
        self._make_flattened_feature_name_image()
        pass

    def transform(self, features):
        '''
        summary
            Takes in features (X), NOT LABELS, and creates an
            instance of a pipeline that takes in the features X
            and returns a DataFrame of image features out of it.
        parameters
            features: a DataFrame in typical data science format 
            (rows are examples, columns are features)
        returns
            a DataFrame in the image format (vector of #pixels*#channels)
        '''
        pass
    def fit_and_transform(self, features):
        '''
        summary
            Performs the fit and transform operations on the features,
            and then returns the result of the transform operation.
        parameters
            features: a DataFrame in typical data science format 
            (rows are examples, columns are features)
        returns
            a DataFrame in the image format (vector of #pixels*#channels)
        '''
        self.fit(features)
        return self.transform(features)

if __name__ == '__main__':
    n_rows = 100
    categories_1 = ('a', 'b', 'c', 'd', 'e')
    categories_2 = ('InstaMed', 'is', 'a', 'cool', 'company', 'check', 'it', 'out', 'sometime')
    rand_cats_1 = [rd.choice(categories_1) for i in range(n_rows)]
    rand_cats_2 = [rd.choice(categories_2) for i in range(n_rows)]
    example_df = pd.DataFrame({
        'example_numerical_col_1': np.random.rand(n_rows) * 50,
        'example_numerical_col_2': np.random.rand(n_rows) * 20,
        'example_categorical_col_1': rand_cats_1,
        'example_categorical_col_2': rand_cats_2,
        'example_boolean_col_1': np.random.randint(low=0, high=2, size=n_rows),
        'example_boolean_col_2': np.random.randint(low=0, high=2, size=n_rows)
    })
    example_feature_names = FeatureTypes(
        categorical=['example_categorical_col_1', 'example_categorical_col_2'],
        numerical=['example_numerical_col_1', 'example_numerical_col_2'],
        boolean=['example_boolean_col_1', 'example_boolean_col_2']
    )
    image16by16by3 = ImageFromTable(example_feature_names, image_side_len=4)
    image16by16by3.fit(example_df)
