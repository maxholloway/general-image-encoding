class FeatureTypes:
    def __init__(self, categorical=[], numerical=[], date=[], boolean=[]):
        ''' Sets specified instance variables. Each variable should be a list of strings.'''
        self.Categorical = categorical
        self.Numerical = numerical
        self.Date = date
        self.Boolean = boolean

class SingleFeatureOperations:
    Relu = 'relu'
    Sigmoid = 'sigmoid'
    
class TwoFeatureOperations:
    Add = 'add'
    Subtract = 'subtract'
    Multiply = 'multiply'

class BooleanOperations:
    And = 'AND'
    Or = 'OR'
    Xor = 'XOR'
    Nand = 'NAND'


import pandas as pd
import numpy as np
import random as rd
import math


class SynthesizeFeatures:
    def __init__(self, feature_names, image_side_len=16, channels_per_pixel=3):
        ''' 
        Set instance variables. Defaults to a 16x16 image with three pixel
        channels, such as with a 16x16 RGB image.
        '''
        # TODO: build in type checking; features_names must be of type FeatureTypes
        self._level_to_features = dict()
        self._level_to_features[0] = feature_names # level 0 features
        self._total_num_features = (image_side_len ** 2) * channels_per_pixel
        print(f'Aiming to have a total of {self._total_num_features} features.')
        assert SynthesizeFeatures._valid_image_side_len(image_side_len), "Image side length is invalid."
        
    @staticmethod
    def _valid_image_side_len(image_side_len, max_pixels=4096):
        ''' 
        In order to be valid, the total number of pixels must
        be a power of four. This is because the feature placement
        algorithm divides the image into quadrants, subquadrants,
        subsubquadrants, ... to construct an image from the features.
        
        Optional pixel_threshold argument specifies maximum number of pixels;
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
    
    def synthesize_features(self, df_inp):
        df = df_inp.copy()

        def completed(num_feats_to_create):
            return num_feats_to_create <= 0

        def concat_new_features(old_df, new_features_inp):
            new_features_dfs = [new_features_inp[key] for key in new_features_inp.keys()]
            new_feature_lens = [len(nf.columns) for nf in new_features_dfs]
            print(f'Going to have {len(old_df.columns) + sum(new_feature_lens)} features after concat.')
            new_features_dfs.append(old_df)
            return pd.concat( new_features_dfs, axis=1)
        
        numerical_summary_stats = ('mean', 'std', 'skew', 'median')
        single_feature_operations = ('log', 'exp', 'sin', 'relu', 'sigmoid')
        two_numerical_feature_operations = ('add', 'multiply', 'subtract')
        boolean_operations = ('AND', 'OR', 'XOR', 'NAND')

        num_feats_to_create = self._total_num_features - len(df_inp.columns)
        print(f'Going to create {num_feats_to_create} synthetic features')
        level = 0
        while True:
            feature_names = self._level_to_features[level]
            new_features = {
                'num' : pd.DataFrame(index=df.index),
                'cat' : pd.DataFrame(index=df.index),
                'bool' : pd.DataFrame(index=df.index),
                'date' : pd.DataFrame(index=df.index)
            }
            
            # Populate with synthetic features related to groups and categorical variables
            for cat_feat_name in  feature_names.Categorical:
                category_groups = df.groupby(cat_feat_name)
                for num_feat_name in feature_names.Numerical:
                    for summary_stat in numerical_summary_stats:
                        if completed(num_feats_to_create): 
                            return concat_new_features(df, new_features)
                        new_feature_name = f'group|{cat_feat_name}|num|{num_feat_name}|op|{summary_stat}|'
                        new_feature = category_groups[num_feat_name].transform(summary_stat)
                        new_features['num'][new_feature_name] =  new_feature
                        num_feats_to_create -= 1

                # create mode of other categorical variables, while in this category's group
                # this is currently untested, and needs to be tested before uncommenting
                for other_cat_feat_name in feature_names.Categorical:
                    if cat_feat_name == other_cat_feat_name: continue
                    #if completed(num_feats_to_create):
                     #   return concat_new_features(df, new_features)
                    #new_feature_name = f'group|{cat_feat_name}|cat|{other_cat_feat_name}|op|mode|'
                    #new_feature = category_groups[other_cat_feat_name].agg( lambda x: pd.Series.mode(x)[0] )
                    #new_features['cat'][new_feature_name] = new_feature
                    #num_feats_to_create -= 1

            # Populate with synthetic features related to numerical transformations
            for i, num_feat_name in enumerate(feature_names.Numerical):
                for single_feat_op in single_feature_operations:
                    if completed(num_feats_to_create): 
                        return concat_new_features(df, new_features) 
                    new_feature_name = f'num|{num_feat_name}|op|{single_feat_op}|'
                    new_feature = SynthesizeFeatures._apply_single_num_feat_operation(df[num_feat_name], single_feat_op)
                    new_features['num'][new_feature_name] = new_feature
                    num_feats_to_create -= 1
                    
                other_num_feature_names = feature_names.Numerical[i+1:] # avoids pairs of features from getting called twice; also avoids pairing with self
                for other_num_feat_name in other_num_feature_names:
                    for two_feature_operation in two_numerical_feature_operations:
                        if completed(num_feats_to_create): 
                            return concat_new_features(df, new_features)
                        new_feature_name = f'num1|{num_feat_name}|num2|{other_num_feat_name}|op|{two_feature_operation}|'
                        new_feature = SynthesizeFeatures._apply_two_num_feat_operation(df[num_feat_name], df[other_num_feat_name], two_feature_operation)
                        new_features['num'][new_feature_name] = new_feature
                        num_feats_to_create -= 1
            
            # Populate with synthetic features related to boolean transformations
            for i, bool_feat_name in enumerate(feature_names.Boolean):
                other_bool_feat_names = feature_names.Boolean[i+1:]
                for other_bool_feat_name in other_bool_feat_names:
                    for bool_op in boolean_operations:
                        if completed(num_feats_to_create):
                            return concat_new_features(df, new_features)
                        new_feature_name = f'bool_feat1|{bool_feat_name}|bool_feat2|{other_bool_feat_name}|op|{bool_op}|'
                        new_feature = SynthesizeFeatures._apply_two_bool_feat_operation(df[bool_feat_name], df[other_bool_feat_name], bool_op)
                        new_features['bool'][new_feature_name] = new_feature
                        num_feats_to_create -= 1

            # Update df and _level_to_features before looping
            print(f'DF num features: {len(df.columns)}')
            df = concat_new_features(df, new_features)
            assert len(df.columns) + num_feats_to_create == self._total_num_features, "Size mismatch; features not appended correctly"
            print(f'DF num features: {len(df.columns)}')
            self._level_to_features[level+1] = FeatureTypes( 
                numerical = new_features['num'].columns,
                categorical = new_features['cat'].columns,
                boolean = new_features['bool'].columns,
                date = new_features['date'].columns)
            level += 1    
    
    @staticmethod
    def _apply_single_num_feat_operation(feature, operation_str):
        _builtin_operation_strs = ('exp', 'log', 'sin', 'cos', 'tan', 
                                   'sinh', 'cosh', 'tanh')
        if operation_str in _builtin_operation_strs:
            return feature.apply(operation_str) # not safe, because may cause problems if out of range
        else:
            if operation_str == SingleFeatureOperations.Relu:
                func = lambda x : 0 if x <= 0 else x
            elif operation_str == SingleFeatureOperations.Sigmoid:
                func = lambda x : 1 / (1 +  math.exp(-x))
            return feature.apply(func)
    
    @staticmethod
    def _apply_two_num_feat_operation(feat1, feat2, operation_str):
        if operation_str == TwoFeatureOperations.Add:
            return feat1 + feat2
        elif operation_str == TwoFeatureOperations.Subtract:
            return feat1-feat2
        elif operation_str == TwoFeatureOperations.Multiply:
            return feat1 * feat2

    @staticmethod
    def _apply_two_bool_feat_operation(feature1, feature2, operation_str):
        combined = pd.DataFrame()
        combined['x1'] = feature1
        combined['x2'] = feature2
        def xor_func(x):
            a, b = x[0], x[1]
            return True if ((a and not b) or (b and not a)) else False
        def nand_func(x):
            a, b = x[0], x[1]
            return not (a and b)
        def or_func(x):
            a, b = x[0], x[1]
            return a or b
        def and_func(x):
            a, b = x[0], x[1]
            return a and b

        if operation_str == BooleanOperations.Xor:
            func = xor_func
        elif operation_str == BooleanOperations.And:
            func = and_func
        elif operation_str == BooleanOperations.Or:
            func = or_func
        elif operation_str == BooleanOperations.Nand:
            func = nand_func

        return combined.apply(lambda x : func(x), axis=1)


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

    sf = SynthesizeFeatures(example_feature_names, image_side_len=16, channels_per_pixel=3)
    new_feats = sf.synthesize_features(example_df)

    print(new_feats.head())


