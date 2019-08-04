import pandas as pd
import numpy as np
import random as rd
import math
import json

class FeatureTypes:
    def __init__(self, categorical=[], numerical=[], date=[], boolean=[]):
        ''' Sets specified instance variables. Each variable should be a list of strings.'''
        self.Categorical = categorical
        self.Numerical = numerical
        self.Date = date
        self.Boolean = boolean

class FeatureDataTypes:
    Categorical = 'categorical'
    Numerical = 'numerical'
    Boolean = 'bool'
    Date = 'date'

class IncorrectTypeException(Exception):
    def __init__(self, variable_name, expected_type, actual_type):
        message = f'Expected type of {variable_name} to be {expected_type}' \
                      f', but it was actually of type {actual_type}.'
        super(IncorrectTypeException, self).__init__(message)
        
class SingleFeatureOperations:
    Relu = 'relu'
    Sigmoid = 'sigmoid'
    Square = 'square'
    Cube = 'cube'
    
class TwoFeatureOperations:
    Add = 'add'
    Subtract = 'subtract'
    Multiply = 'multiply'
    Square = 'square'
    Cube = 'cube'

class BooleanOperations:
    And = 'AND'
    Or = 'OR'
    Xor = 'XOR'
    Nand = 'NAND'

class FeatureHandlingMethods:
    ''''''
    ImputeMedian = 'imputeMedian'
    ImputeMean = 'imputeMean'
    Zero = 'zero'
    Remove = 'remove'

class FeatureStates:
    ''''''
    Good = 'noNullValues'
    Ok = 'someNullValues'   
    Bad = 'tooManyNullValues'

class FeatureNameObject:
    def __init__(self, new_feature_type, old_feature_names, operation):
        '''
        :param new_feature_type:
        :param old_feature_names:
        :param operation: string of the operation to be performed on the old feature names
        '''
        self.n = new_feature_type
        self.o = old_feature_names
        self.p = operation

class FeatureOperations:
    ''''''

    def __init__(self, max_proportion_null, handling_method):
        self.max_proportion_null = max_proportion_null
        self.handling_method = handling_method
        return

    def try_add_new_categorical_summary_stat_feat_op_feature(self, groups, new_features, operation, cat_feat_name, num_feat_name):
        '''
        :param group:
        :param new_features:
        :param summary_stat:
        :param cat_feat_name:
        :param num_feat_name:
        :return:
        '''
        new_feature = groups[num_feat_name].transform(operation)
        self.__handle_feature(new_feature, new_features, FeatureDataTypes.Numerical, [num_feat_name, cat_feat_name], operation)
        return

    def try_add_new_single_num_feat_op_feature(self, df, new_features, single_feat_op, num_feat_name):
        '''
        :param df: pandas.DataFrame, where each column is a feature, each row is an example
        :param new_features: dictionary mapping from feature type (e.g. categorical) to a pandas.DataFrame,
                             where the columns are features of that feature type
        :param single_feat_op: a string that indicates the type of single feature operation to perform
        :param num_feat_name: name of the column that has the numerical feature to be transformed
        :return: None; this method edits new_features in-place (to avoid cost of copying variables)
        '''
        new_feature = self.__apply_single_num_feat_operation(df[num_feat_name], single_feat_op)
        self.__handle_feature(new_feature, new_features, FeatureDataTypes.Numerical, [num_feat_name], single_feat_op)
        return

    def try_add_new_dual_num_feat_op_feature(self, df, new_features, operation, num_feat_name1, num_feat_name2):
        '''
        :param df:
        :param new_features:
        :param dual_feat_op:
        :param num_feat_name1:
        :param num_feat_name2:
        :return:
        '''
        new_feature = self.__apply_two_num_feat_operation(df[num_feat_name1], df[num_feat_name2], operation)
        self.__handle_feature(new_feature, new_features, FeatureDataTypes.Numerical, [num_feat_name1, num_feat_name2],
                              operation)
        return

    def try_add_new_dual_bool_feat_op_feature(self, df, new_features, operation, bool_feat_name1, bool_feat_name2):
        '''
        :param new_features:
        :param operation:
        :param bool_feat_name:
        :param other_bool_feat_name:
        :return:
        '''
        new_feature = self.__apply_two_bool_feat_operation(df[bool_feat_name1], df[bool_feat_name2], operation)
        self.__handle_feature(new_feature, new_features, FeatureDataTypes.Boolean, [bool_feat_name1, bool_feat_name2],
                              operation)
        return

    def __get_feature_state(self, feature):
        '''
        summary
            self explanatory
        parameters
            feature: pandas.Series
        returns
            the state of the feature (one of the class variables)
        '''
        if type(feature) != type(pd.Series()):
            raise IncorrectTypeException('feature', type(pd.Series()), type(feature))
        else:
            num_null = feature.isnull().sum()
            prop_null = num_null / feature.size
            if prop_null > self.max_proportion_null:
                return FeatureStates.Bad
            elif prop_null > 0:
                return FeatureStates.Ok
            else:
                return FeatureStates.Good
    
    def __handle_feature(self, feature, new_features, feat_data_type, old_feat_names, operation, verbose=False):
        '''
        summary
            This method checks feature to see if it complies with standards
            for features given by instance variables, then performs any basic data
            cleaning necessary for it to be added to new_features.
        :param feature:
        :param new_features:
        :param feat_data_type:
        :param new_feat_names: list of strings
        :param operation: string
        :param verbose: bool
        :return: None; either edits new_features in-place, or if feature is too bad, it makes no state changes at all
        '''

        '''

        parameters
            feature: pandas.Series
        returns
            pandas.Series of the new feature, unless the feature is ruled as bad, in which case
            the value None will be returned.
        '''
        if type(feature) != type(pd.Series()):
            raise IncorrectTypeException('feature', type(pd.Series()), type(feature))

        feature_state = self.__get_feature_state(feature)
        if feature_state == FeatureStates.Good:
            pass # do nothing, because the feature is already perfect
        elif feature_state == FeatureStates.Ok:
            if self.handling_method == FeatureHandlingMethods.ImputeMedian:
                feature = feature.fillna(feature.median())
            elif self.handling_method == FeatureHandlingMethods.ImputeMean:
                feature = feature.fillna(feature.mean())
            elif self.handling_method == FeatureHandlingMethods.Zero:
                feature = feature.fillna(0)
            elif self.handling_method == FeatureHandlingMethods.Remove:
                return # don't add feature
            else:
                raise Exception('Invalid handling method; check initialization of this instance'
                                + ' of FeatureOperations; it may help to use a FeatureHandlingMethods'
                               + 'class variable, to avoid typographic mistakes.')
        elif feature_state == FeatureStates.Bad:
            if verbose:
                print('from FeatureOperations.handle_features: A feature was labeled as "tooManyNullValues", or'
                 + 'FeatureStates.Bad. Returning None')
            return
        else:
            raise Exception("Invalid Feature State")

        # if method execution gets to here, the feature has been handled and is in OK condition to add to new_features
        new_feature_name = self.__make_feature_name(feat_data_type, old_feat_names, operation)
        new_features[feat_data_type][new_feature_name] = feature
        return

    def __make_feature_name(self, new_feature_type, old_feature_names, operation):
        '''
        :param new_feature_type: string
        :param old_feature_names: list of strings
        :param operation: string giving the name of the operation to be performed
        :return: a string that gives the new feature name
        '''
        feature_name_obj = FeatureNameObject(new_feature_type, old_feature_names, operation)
        return json.dumps(feature_name_obj.__dict__, separators=(',', ':'), indent=None)

    def __apply_single_num_feat_operation(self, feature, operation_str):
        builtin_operation_strs = ('exp', 'log', 'sin', 'cos', 'tan', 
                                   'sinh', 'cosh', 'tanh')
        if operation_str in builtin_operation_strs:
            return feature.apply(operation_str) # not safe, because may cause problems if out of range
        else:
            if operation_str == SingleFeatureOperations.Relu:
                func = lambda x : 0 if x <= 0 else x
            elif operation_str == SingleFeatureOperations.Sigmoid:
                func = lambda x : 1 / (1 +  math.exp(-x))
            elif operation_str == SingleFeatureOperations.Square:
                func = lambda x : x**2
            elif operation_str == SingleFeatureOperations.Cube:
                func = lambda x : x**3
            return feature.apply(func)
    
    def __apply_two_num_feat_operation(self, feat1, feat2, operation_str):
        if operation_str == TwoFeatureOperations.Add:
            return feat1 + feat2
        elif operation_str == TwoFeatureOperations.Subtract:
            return feat1-feat2
        elif operation_str == TwoFeatureOperations.Multiply:
            return feat1 * feat2

    def __apply_two_bool_feat_operation(self, feature1, feature2, operation_str):
        combined = pd.DataFrame()
        combined['x1'] = feature1
        combined['x2'] = feature2
        def xor_func(x):
            a, b = x[0], x[1]
            return 1 if ((a and not b) or (b and not a)) else 0
        def nand_func(x):
            a, b = x[0], x[1]
            return int(not (a and b))
        def or_func(x):
            a, b = x[0], x[1]
            return int(a or b)
        def and_func(x):
            a, b = x[0], x[1]
            return int(a and b)

        if operation_str == BooleanOperations.Xor:
            func = xor_func
        elif operation_str == BooleanOperations.And:
            func = and_func
        elif operation_str == BooleanOperations.Or:
            func = or_func
        elif operation_str == BooleanOperations.Nand:
            func = nand_func

        new_feature = combined.apply(lambda x : func(x), axis=1) 
        if type(new_feature) != type(pd.Series()):
            raise IncorrectTypeException('new_feature', type(pd.Series()), type(new_feature))
        return new_feature

class FeatureSynthesis:
    _DEFAULT_FEATURE_HANDLING_METHOD = FeatureHandlingMethods.ImputeMedian
    _DEFAULT_MAX_PROPORTION_NULL_VALUES = 0.1
    
    def __init__(self, feature_names, total_num_features):
        ''' 
        Set instance variables. Defaults to a 16x16 image with three pixel
        channels, such as with a 16x16 RGB image.
        '''
        # TODO: build in type checking; features_names must be of type FeatureTypes
        self._level_to_features = dict()
        self._level_to_features[0] = feature_names # level 0 features
        self._total_num_features = total_num_features
        #print(f'Aiming to have a total of {self._total_num_features} features.')
        
    def synthesize_features(self, df_inp, max_prop_null=_DEFAULT_MAX_PROPORTION_NULL_VALUES, 
                            null_handling_method=_DEFAULT_FEATURE_HANDLING_METHOD):
        df = df_inp.copy()
        
        def _num_features_helper(old_df, new_features_inp):
            new_features_dfs = [new_features_inp[key] for key in new_features_inp.keys()]
            new_feature_lens = [len(nf.columns) for nf in new_features_dfs]
            return len(old_df.columns) + sum(new_feature_lens)

        def completed(df_inp, new_feats_inp):
            num_new_features = _num_features_helper(df_inp, new_feats_inp)
            return num_new_features >= self._total_num_features

        def _update_level_to_features(new_features):
            '''
            '''
            if level+1 not in self._level_to_features.keys():
                self._level_to_features[level+1] = FeatureTypes( 
                    numerical = list(new_features[FeatureDataTypes.Numerical].columns),
                    categorical = list(new_features[FeatureDataTypes.Categorical].columns),
                    boolean = list(new_features[FeatureDataTypes.Boolean].columns),
                    date = list(new_features[FeatureDataTypes.Date].columns))
            else: 
                self._level_to_features[level+1].Numerical += list(new_features[FeatureDataTypes.Numerical].columns.values)
                self._level_to_features[level+1].Categorical += list(new_features[FeatureDataTypes.Categorical].columns.values)
                self._level_to_features[level+1].Date += list(new_features[FeatureDataTypes.Date].columns.values)
                self._level_to_features[level+1].Boolean += list(new_features[FeatureDataTypes.Boolean].columns.values)
        
        def concat_new_features(old_df, new_features_inp):
            new_features_dfs = [new_features_inp[key] for key in new_features_inp.keys()]
            new_feature_lens = [len(nf.columns) for nf in new_features_dfs]
            #print(f'Going to have {len(old_df.columns) + sum(new_feature_lens)} features after concat.')
            new_features_dfs.append(old_df)
            _update_level_to_features(new_features)
            return pd.concat( new_features_dfs, axis=1)
        
        numerical_summary_stats = ('mean', 'std', 'skew', 'median')
        single_feature_operations = ('log', 'exp', 'sin', 'relu', 'sigmoid', 'square', 'cube')
        two_numerical_feature_operations = ('add', 'multiply', 'subtract')
        boolean_operations = ('AND', 'OR', 'XOR', 'NAND')

        feat_ops = FeatureOperations(max_prop_null, null_handling_method)

        level = 0
        while True:
            feature_names = self._level_to_features[level]
            new_features = {
                FeatureDataTypes.Numerical : pd.DataFrame(index=df.index),
                FeatureDataTypes.Categorical : pd.DataFrame(index=df.index),
                FeatureDataTypes.Boolean: pd.DataFrame(index=df.index),
                FeatureDataTypes.Date : pd.DataFrame(index=df.index)
            }
            
            # Populate with synthetic features related to groups and categorical variables
            for cat_feat_name in  feature_names.Categorical:
                category_groups = df.groupby(cat_feat_name)
                for num_feat_name in feature_names.Numerical:
                    for summary_stat in numerical_summary_stats:
                        if completed(df, new_features):
                            return concat_new_features(df, new_features)
                        feat_ops.try_add_new_categorical_summary_stat_feat_op_feature(category_groups, new_features,
                                                                                      summary_stat, cat_feat_name,
                                                                                      num_feat_name)

                # create mode of other categorical variables, while in this category's group
                # this is currently untested, and needs to be tested before uncommenting
                for other_cat_feat_name in feature_names.Categorical:
                    if cat_feat_name == other_cat_feat_name: continue
                    #if completed(num_feats_to_create): 
                     #   return concat_new_features(df, new_features)
                    #new_feature_name = f'group|{cat_feat_name}|cat|{other_cat_feat_name}|op|mode|'
                    #new_feature = category_groups[other_cat_feat_name].agg( lambda x: pd.Series.mode(x)[0] )
                    #new_features['cat'][new_feature_name] = new_feature

            # Populate with synthetic features related to numerical transformations
            for i, num_feat_name in enumerate(feature_names.Numerical):
                for single_feat_op in single_feature_operations:
                    if completed(df, new_features):
                        return concat_new_features(df, new_features)
                    feat_ops.try_add_new_single_num_feat_op_feature(df, new_features, single_feat_op, num_feat_name)

                other_num_feature_names = feature_names.Numerical[i+1:] # avoids pairs of features from getting called twice; also avoids pairing with self
                for other_num_feat_name in other_num_feature_names:
                    for dual_feat_op in two_numerical_feature_operations:
                        if completed(df, new_features):
                            return concat_new_features(df, new_features)
                        feat_ops.try_add_new_dual_num_feat_op_feature(df, new_features, dual_feat_op,
                                                                      num_feat_name, other_num_feat_name)

            # Populate with synthetic features related to boolean transformations
            for i, bool_feat_name in enumerate(feature_names.Boolean):
                other_bool_feat_names = feature_names.Boolean[i+1:]
                for other_bool_feat_name in other_bool_feat_names:
                    for bool_op in boolean_operations:
                        if completed(df, new_features):
                            return concat_new_features(df, new_features)
                        feat_ops.try_add_new_dual_bool_feat_op_feature(df, new_features, bool_op,
                                                                       bool_feat_name, other_bool_feat_name)

            # TODO: Create Date and Categorical synthetic features
            
            # Update df and _level_to_features before looping
            #print(f'DF num features: {len(df.columns)}')
            df = concat_new_features(df, new_features)
            #print(f'DF num features: {len(df.columns)}')
            self._level_to_features[level+1] = FeatureTypes( 
                numerical = list(new_features[FeatureDataTypes.Numerical].columns.values),
                categorical = list(new_features[FeatureDataTypes.Categorical].columns.values),
                boolean = list(new_features[FeatureDataTypes.Boolean].columns.values),
                date = list(new_features[FeatureDataTypes.Date].columns.values))
            level += 1   
        
        return
        
    def get_col_names(self, feature_data_type):
        '''
        TODO: Test this further
        summary
            Gets names of columns of the specified data type
        parameters
            feature_data_type: string (one of the FeatureDataTypes instance variables)
        returns
            list of strings with the names of the columns of the specified data type
        '''
        levels = list(self._level_to_features.keys())
        if feature_data_type == FeatureDataTypes.Numerical:
            col_names_lists = [self._level_to_features[level].Numerical for level in levels]
        elif feature_data_type == FeatureDataTypes.Categorical:
            col_names_lists = [self._level_to_features[level].Categorical for level in levels]
        elif feature_data_type == FeatureDataTypes.Date:
            col_names_lists = [self._level_to_features[level].Date for level in levels]
        elif feature_data_type == FeatureDataTypes.Boolean:
            col_names_lists = [self._level_to_features[level].Boolean for level in levels]
        else:
            raise Exception('"feature_data_type" was invalid; should have been equal to one of the '
                           + f'FeatureDataTypes instance variables, but instead was {feature_data_type}')
        all_col_names = []
        for col_name_list in col_names_lists:
            #print(f'Trying to append {col_name_list} onto {all_col_names}')
            all_col_names += col_name_list
        return all_col_names

   

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

    import time
    t0 = time.time()
    fs = FeatureSynthesis(example_feature_names, total_num_features=768) #768
    new_feats = fs.synthesize_features(example_df)
    t1 = time.time()
    print(new_feats.head())
    print(f'Took {t1-t0} seconds to finish.')

