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

    @staticmethod
    def get_all_feature_data_types():
        return (FeatureDataTypes.Categorical, FeatureDataTypes.Numerical, FeatureDataTypes.Boolean, FeatureDataTypes.Date)

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

class FeatureNullityStates:
    ''''''
    Good = 'noNullValues'
    Ok = 'someNullValues'   
    Bad = 'tooManyNullValues'

class FeatureName:
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

    def try_add_new_categorical_summary_stat_feat_op_feature(self, groups, all_features, operation, cat_feat_name,
                                                             num_feat_name):
        '''
        :param group:
        :param all_features:
        :param summary_stat:
        :param cat_feat_name:
        :param num_feat_name:
        :return:
        '''
        new_feature = groups[num_feat_name].transform(operation)
        self.__handle_feature(new_feature, all_features, FeatureDataTypes.Numerical, [num_feat_name, cat_feat_name], operation)
        return

    def try_add_new_single_num_feat_op_feature(self, df, all_features, single_feat_op, num_feat_name):
        '''
        :param df: pandas.DataFrame, where each column is a feature, each row is an example
        :param all_features: AllFeatures object
        :param single_feat_op: a string that indicates the type of single feature operation to perform
        :param num_feat_name: name of the column that has the numerical feature to be transformed
        :return: None; this method edits new_features in-place (to avoid cost of copying variables)
        '''
        new_feature = self.__apply_single_num_feat_operation(df[num_feat_name], single_feat_op)
        self.__handle_feature(new_feature, all_features, FeatureDataTypes.Numerical, [num_feat_name], single_feat_op)
        return

    def try_add_new_dual_num_feat_op_feature(self, df, all_features, operation, num_feat_name1, num_feat_name2):
        '''
        :param df:
        :param new_features:
        :param dual_feat_op:
        :param num_feat_name1:
        :param num_feat_name2:
        :return:
        '''
        new_feature = self.__apply_two_num_feat_operation(df[num_feat_name1], df[num_feat_name2], operation)
        self.__handle_feature(new_feature, all_features, FeatureDataTypes.Numerical, [num_feat_name1, num_feat_name2],
                              operation)
        return

    def try_add_new_dual_bool_feat_op_feature(self, df, all_features, operation, bool_feat_name1, bool_feat_name2):
        '''
        :param new_features:
        :param operation:
        :param bool_feat_name:
        :param other_bool_feat_name:
        :return:
        '''
        new_feature = self.__apply_two_bool_feat_operation(df[bool_feat_name1], df[bool_feat_name2], operation)
        self.__handle_feature(new_feature, all_features, FeatureDataTypes.Boolean, [bool_feat_name1, bool_feat_name2],
                              operation)
        return

    def __get_feature_nullity_state(self, feature):
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

        num_null = feature.isnull().sum()
        prop_null = num_null / feature.size
        if prop_null > self.max_proportion_null:
            return FeatureNullityStates.Bad
        elif prop_null > 0:
            return FeatureNullityStates.Ok
        else:
            return FeatureNullityStates.Good

    @staticmethod
    def __feature_is_too_uniform(feature, max_prop_same=1.0):
        '''
        summary:
            Checks if feature values are uniform.
        :param feature: pandas.Series feature
        :param max_prop_same: float in range [0, 1]; determines
                              proportion of values that can be the
                              same without discarding feature as
                              giving no unique information; bound
                              is exclusive (so value of x will discard
                              any features with prop_same >= x)
        :return: bool that is True if the feature values are uniform,
                 False otherwise
        '''
        try:
            feature_first_el = feature.values[0]
            same_vals = feature.values == feature_first_el
            num_same_vals = np.sum(same_vals)
            total_num_vals = feature.size
            if num_same_vals/total_num_vals >= max_prop_same:
                return True
            else:
                return False
        except Exception:
            print('An error occurred in __feature_is_uniform(), and it\'s unclear how to handle it.\n'
                  + 'Erring on the side of caution and returning True.')
            return True

    @staticmethod
    def __new_feature_is_too_similar_to_existing_features(new_feature, existing_features, max_prop_same=1.0):
        '''
        :param new_feature: pandas.Series
        :param existing_features: pandas.DataFrame
        :param max_prop_same: the maximum proportion of elements that can be shared between
                              the new_feature and any given column in existing_features; defaults
                              to 1.0, meaning it will only return True if the entire feature is a duplicate
        :return: bool that is True if the given feature is already a column in
                 the DataFrame (disregarding column names), and False otherwise
        '''
        existing_feature_vals = existing_features.values
        num_rows = new_feature.shape[0]
        new_feature_vals = np.reshape(new_feature.values, (num_rows, 1))
        column_equals_feature = existing_feature_vals == new_feature_vals
        num_same_each_column = np.sum(column_equals_feature, axis=0)
        prop_same_each_column = num_same_each_column / num_rows
        cols_entirely_same = (prop_same_each_column == 1)
        if np.sum(cols_entirely_same) > 1:
            raise Exception('Previously there was a feature added that should not have been added (or perhaps some of '
                            +'the input features were duplicates, which is also no-bueno).')
        cols_too_similar = (prop_same_each_column >= max_prop_same)
        if(np.sum(cols_too_similar) >= 1):
            return True
        else:
            return False

    def __is_redundant_feature(self, all_features, new_feature):
        '''
        summary:
            Checks if new_feature is a duplicate of an already existing
            feature, or if new_feature gives no information (for example,
            new_feature is always the same value).
        :param all_features: AllFeatures object
        :param new_feature: pandas.Series
        :return: bool False if feature is too uniform or too similar to other existing features, else False
        '''

        existing_features = all_features.concat_new_features()

        if FeatureOperations.__feature_is_too_uniform(new_feature) \
          or FeatureOperations.__new_feature_is_too_similar_to_existing_features(new_feature,
                                                                                 existing_features):
            return True
        else:
            return False

    def __handle_feature(self, feature, old_features, feat_data_type, old_feat_names, operation, verbose=False):
        '''
        summary
            This method checks feature to see if it complies with standards
            for features given by instance variables, then performs any basic data
            cleaning necessary for it to be added to new_features.
        :param feature: pandas.Series
        :param old_features: AllFeatures object
        :param feat_data_type: string
        :param old_feat_names: list of strings
        :param operation: string
        :param verbose: bool
        :return: None; either edits new_features in-place, or if feature is too bad, it makes no state changes at all
        '''
        if type(feature) != type(pd.Series()):
            raise IncorrectTypeException('feature', type(pd.Series()), type(feature))

        # Check for nullity (is the feature too empty?)
        feature_nullity_state = self.__get_feature_nullity_state(feature)
        if feature_nullity_state == FeatureNullityStates.Good:
            pass # do nothing, because the feature is already perfect
        elif feature_nullity_state == FeatureNullityStates.Ok:
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
        elif feature_nullity_state == FeatureNullityStates.Bad:
            if verbose:
                print('from FeatureOperations.handle_features: A feature was labeled as "tooManyNullValues", or'
                 + 'FeatureStates.Bad. Returning None')
            return
        else:
            raise Exception("Invalid Feature State")

        if not self.__is_redundant_feature(old_features, feature):
            # if method execution gets to here, the feature has been handled and is in OK condition to add to new_features
            new_feature_name = self.__make_feature_name(feat_data_type, old_feat_names, operation)
            old_features.add_feature(feature, feat_data_type, new_feature_name)
        return

    def __make_feature_name(self, new_feature_type, old_feature_names, operation):
        '''
        :param new_feature_type: string
        :param old_feature_names: list of strings
        :param operation: string giving the name of the operation to be performed
        :return: a string that gives the new feature name
        '''
        feature_name_obj = FeatureName(new_feature_type, old_feature_names, operation)
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

class AllFeatures:
    def __init__(self, total_num_features, original_features, original_feature_names):
        self.__old_features = original_features.copy()
        self.__index = original_features.index.copy()
        self.__total_num_features = total_num_features
        self.__level = 0
        self.__level_to_features = dict()
        self.__level_to_features[self.__level] = original_feature_names # of type FeatureTypes

        return

    def set_new_features(self):
        self.__new_features = {
            FeatureDataTypes.Numerical: pd.DataFrame(index=self.__index),
            FeatureDataTypes.Categorical: pd.DataFrame(index=self.__index),
            FeatureDataTypes.Boolean: pd.DataFrame(index=self.__index),
            FeatureDataTypes.Date: pd.DataFrame(index=self.__index)
        }
        return

    def __num_features_helper(self):
        new_features_dfs = [self.__new_features[dtype] for dtype in self.__new_features.keys()]
        new_feature_lens = [nf.shape[1] for nf in new_features_dfs]
        return self.__old_features.shape[1] + sum(new_feature_lens)

    def completed(self):
        num_new_features = self.__num_features_helper()
        if num_new_features >= self.__total_num_features:
            return True
        else:
            return False

    def __update_level_to_features(self):
        '''
        '''
        if self.__level + 1 not in self.__level_to_features.keys():
            self.__level_to_features[self.__level + 1] = FeatureTypes(
                numerical=list(self.__new_features[FeatureDataTypes.Numerical].columns),
                categorical=list(self.__new_features[FeatureDataTypes.Categorical].columns),
                boolean=list(self.__new_features[FeatureDataTypes.Boolean].columns),
                date=list(self.__new_features[FeatureDataTypes.Date].columns))
            pass
        else:
            raise Exception(f'Unexpected behavior in "__update_level_to_features()"')

    def get_new_features_df(self):
        return pd.concat([self.__new_features[key] for key in self.__new_features.keys()], axis=1)

    def concat_new_features(self):
        '''
        :return: pandas.DataFrame of all current features, including self.__old_features and self.__new_features
        '''
        return pd.concat([self.__old_features, self.get_new_features_df()], axis=1)


    def reset(self):
        '''
        summary:
            prepares the class to take on another level of features
        :return: None
        '''
        self.__old_features = self.concat_new_features()
        self.__update_level_to_features()
        self.set_new_features()
        self.__level += 1
        return

    def add_feature(self, feature, feature_data_type, new_feature_name):
        self.__new_features[feature_data_type][new_feature_name] = feature

    def get_this_level_feat_names(self):
        return self.__level_to_features[self.__level]

    def get_col_names(self, feature_data_type):
        '''
        summary
            Gets names of columns of the specified data type
        parameters
            feature_data_type: string (one of the FeatureDataTypes instance variables)
        returns
            list of strings with the names of the columns of the specified data type
        '''
        levels = list(self.__level_to_features.keys())
        if feature_data_type == FeatureDataTypes.Numerical:
            col_names_lists = [self.__level_to_features[level].Numerical for level in levels]
        elif feature_data_type == FeatureDataTypes.Categorical:
            col_names_lists = [self.__level_to_features[level].Categorical for level in levels]
        elif feature_data_type == FeatureDataTypes.Date:
            col_names_lists = [self.__level_to_features[level].Date for level in levels]
        elif feature_data_type == FeatureDataTypes.Boolean:
            col_names_lists = [self.__level_to_features[level].Boolean for level in levels]
        else:
            raise Exception('"feature_data_type" was invalid; should have been equal to one of the '
                           + f'FeatureDataTypes instance variables, but instead was {feature_data_type}')
        all_col_names = []
        for col_name_list in col_names_lists:
            #print(f'Trying to append {col_name_list} onto {all_col_names}')
            all_col_names += col_name_list
        return all_col_names

    def get_all_col_names(self):
        '''
        :return: dictionary column_data_ype : list<string>, where the
                 strings are the names of the columns
        '''
        col_data_types = FeatureDataTypes.get_all_feature_data_types()
        return {col_data_type : self.get_col_names(col_data_type) for col_data_type in col_data_types}

class FeatureSynthesis:
    __DEFAULT_FEATURE_HANDLING_METHOD = FeatureHandlingMethods.ImputeMedian
    __DEFAULT_MAX_PROPORTION_NULL_VALUES = 0.1
    __DEFAULT_TOTAL_NUM_FEATURES = 768

    def __init__(self):
        self.__feature_names = dict() # maps from data type to feature names of that data type
        return

    def synthesize_features(self, original_features, original_feature_names, total_num_features=__DEFAULT_TOTAL_NUM_FEATURES,
                            max_prop_null=__DEFAULT_MAX_PROPORTION_NULL_VALUES,
                            null_handling_method=__DEFAULT_FEATURE_HANDLING_METHOD):

        # TODO: build in type checking; features_names must be of type FeatureTypes

        numerical_summary_stats = ('mean', 'std', 'skew', 'median')
        single_feature_operations = ('log', 'exp', 'sin', 'relu', 'sigmoid', 'square', 'cube')
        two_numerical_feature_operations = ('add', 'multiply', 'subtract')
        boolean_operations = ('AND', 'OR', 'XOR', 'NAND')

        feat_ops = FeatureOperations(max_prop_null, null_handling_method)
        all_feats = AllFeatures(total_num_features=total_num_features, original_features=original_features,
                                original_feature_names=original_feature_names)

        while True:
            feature_names = all_feats.get_this_level_feat_names()
            all_feats.set_new_features()

            # Populate with synthetic features related to groups and categorical variables
            for cat_feat_name in  feature_names.Categorical:
                category_groups = original_features.groupby(cat_feat_name)
                for num_feat_name in feature_names.Numerical:
                    for summary_stat in numerical_summary_stats:
                        if all_feats.completed():
                            feature_array = all_feats.concat_new_features()
                            all_feats.reset()
                            self.set_feature_names(all_feats.get_all_col_names())
                            return feature_array
                        feat_ops.try_add_new_categorical_summary_stat_feat_op_feature(category_groups, all_feats,
                                                                                      summary_stat, cat_feat_name,
                                                                                      num_feat_name)

                # create mode of other categorical variables, while in this category's group
                # this is currently untested, and needs to be tested before uncommenting
                # for other_cat_feat_name in feature_names.Categorical:
                #     if cat_feat_name == other_cat_feat_name: continue
                    #if completed(num_feats_to_create): 
                     #   return concat_new_features(df, new_features)
                    #new_feature_name = f'group|{cat_feat_name}|cat|{other_cat_feat_name}|op|mode|'
                    #new_feature = category_groups[other_cat_feat_name].agg( lambda x: pd.Series.mode(x)[0] )
                    #new_features['cat'][new_feature_name] = new_feature

            # Populate with synthetic features related to numerical transformations
            for i, num_feat_name in enumerate(feature_names.Numerical):
                for single_feat_op in single_feature_operations:
                    if all_feats.completed():
                        feature_array = all_feats.concat_new_features()
                        all_feats.reset()
                        self.set_feature_names(all_feats.get_all_col_names())
                        return feature_array
                    feat_ops.try_add_new_single_num_feat_op_feature(original_features, all_feats, single_feat_op, num_feat_name)

                other_num_feature_names = feature_names.Numerical[i+1:] # avoids pairs of features from getting called twice; also avoids pairing with self
                for other_num_feat_name in other_num_feature_names:
                    for dual_feat_op in two_numerical_feature_operations:
                        if all_feats.completed():
                            feature_array = all_feats.concat_new_features()
                            all_feats.reset()
                            self.set_feature_names(all_feats.get_all_col_names())
                            return feature_array
                        feat_ops.try_add_new_dual_num_feat_op_feature(original_features, all_feats, dual_feat_op,
                                                                      num_feat_name, other_num_feat_name)

            # Populate with synthetic features related to boolean transformations
            for i, bool_feat_name in enumerate(feature_names.Boolean):
                other_bool_feat_names = feature_names.Boolean[i+1:]
                for other_bool_feat_name in other_bool_feat_names:
                    for bool_op in boolean_operations:
                        if all_feats.completed():
                            feature_array = all_feats.concat_new_features()
                            all_feats.reset()
                            self.set_feature_names(all_feats.get_all_col_names())
                            return feature_array
                        feat_ops.try_add_new_dual_bool_feat_op_feature(original_features, all_feats, bool_op,
                                                                       bool_feat_name, other_bool_feat_name)

            # TODO: Create Date and Categorical synthetic features
            

            original_features = all_feats.get_new_features_df()
            all_feats.reset()

        
        return

    def set_feature_names(self, all_feature_names):
        '''
        :param all_feature_names: dictionary<string, list<string>> from data_type to list of feature names
        :return:
        '''
        self.__feature_names = all_feature_names.copy()
        return

    def get_feature_names(self, data_type):
        '''
        :param data_type: string denoting data type; use FeatureDataTypes class variables for more secure code
        :return: list<string> with feature names
        '''
        return self.__feature_names[data_type]

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
    fs = FeatureSynthesis()
    new_feats = fs.synthesize_features(
        original_features=example_df, original_feature_names=example_feature_names, total_num_features=65)
    t1 = time.time()
    print(new_feats.head())
    print(f'Took {t1-t0} seconds to finish.')

    a = [fs.get_feature_names(dtype) for dtype in FeatureDataTypes.get_all_feature_data_types()]
    for i, el in enumerate(a):
        pd.DataFrame({'feature_name':el}).to_csv(f'{i}.csv')
        print(len(el)) #35 over? must be returning redundantly
    pass

