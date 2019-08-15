import pandas as pd
import numpy as np
import random as rd
import math
import json
from typing import List, Dict, Set


class FeatureTypes:
    def __init__(self, categorical: List[str] = None, numerical: List[str] = None,
                 date: List[str] = None, boolean: List[str] = None):
        """
        :param categorical:
        :param numerical:
        :param date:
        :param boolean:
        """
        self.Categorical = categorical if categorical else []
        self.Numerical = numerical if numerical else []
        self.Date = date if date else []
        self.Boolean = boolean if boolean else []

    def to_dict(self) -> Dict[str, List[str]]:
        """
        Converts the class to a dictionary.
        :return:
        """
        return {'Categorical': self.Categorical, 'Numerical': self.Numerical,
                'Date': self.Date, 'Boolean': self.Boolean}


class FeatureDataTypes:
    Categorical = 'categorical'
    Numerical = 'numerical'
    Boolean = 'bool'
    Date = 'date'

    @staticmethod
    def get_all_feature_data_types():
        return (FeatureDataTypes.Categorical, FeatureDataTypes.Numerical,
                FeatureDataTypes.Boolean, FeatureDataTypes.Date)


class _IncorrectTypeException(Exception):
    def __init__(self, variable_name, expected_type, actual_type):
        message = f'Expected type of {variable_name} to be {expected_type}' \
            f', but it was actually of type {actual_type}.'
        super(_IncorrectTypeException, self).__init__(message)


class _NumericalSummaryStats:
    Mean = 'mean'
    Median = 'median'
    StandardDeviation = 'std'
    Skew = 'skew'

    @staticmethod
    def get_all_operations() -> Set[str]:
        return {
            _NumericalSummaryStats.Mean,
            _NumericalSummaryStats.Median,
            _NumericalSummaryStats.StandardDeviation,
            _NumericalSummaryStats.Skew
        }


class _SingleFeatureOperations:
    Relu = 'relu'
    Sigmoid = 'sigmoid'
    Square = 'square'
    Cube = 'cube'

    @staticmethod
    def get_all_operations() -> Set[str]:
        """
        :return:
        """
        return {
            _SingleFeatureOperations.Relu,
            _SingleFeatureOperations.Sigmoid,
            _SingleFeatureOperations.Square,
            _SingleFeatureOperations.Cube
        }


class _TwoFeatureOperations:
    Add = 'add'
    Subtract = 'subtract'
    Multiply = 'multiply'

    @staticmethod
    def get_all_operations() -> Set[str]:
        return {
            _TwoFeatureOperations.Add,
            _TwoFeatureOperations.Subtract,
            _TwoFeatureOperations.Multiply
        }


class _BooleanOperations:
    """
    Constants used for comparison by other classes.
    """
    And = 'AND'
    Or = 'OR'
    Xor = 'XOR'
    Nand = 'NAND'

    @staticmethod
    def get_all_operations() -> Set[str]:
        return {
            _BooleanOperations.And,
            _BooleanOperations.Or,
            _BooleanOperations.Xor,
            _BooleanOperations.Nand
        }


class _FeatureHandlingMethods:
    """
    Constants used for comparison by other classes.
    """
    ImputeMedian = 'imputeMedian'
    ImputeMean = 'imputeMean'
    Zero = 'zero'
    Remove = 'remove'


class _FeatureNullityStates:
    """
    Constants used for comparison by other classes.
    """
    Good = 'noNullValues'
    Ok = 'someNullValues'
    Bad = 'tooManyNullValues'


class FeatureName:
    def __init__(self, new_feature_type: str, old_feature_names: List[str], operation: str, make_blank: bool = False):
        """
        :param new_feature_type:
        :param old_feature_names:
        :param operation: string of the operation to be performed on the old feature names
        """
        if not make_blank:
            self.new_feat_type = new_feature_type
            self.old_feat_names = old_feature_names
            self.operation = operation
        return

    def serialize_feature_name(self) -> str:
        """
        :return: serialized version of FeatureName
        """
        return json.dumps(self.__dict__, separators=(',', ':'), indent=None)

    @staticmethod
    def deserialize_feature_name(feat_name_str: str):
        """
        Given a serialized feature name string, convert it to a feature name object.
        :param feat_name_str:
        :return:
        """
        feat_name_dict = json.loads(feat_name_str)
        feat_name_obj = FeatureName('', [''], '', True)  # make a blank FeatureName object
        feat_name_obj.__dict__.update(feat_name_dict)
        return feat_name_obj


class AllFeatures:
    def __init__(self, total_num_features: int, original_features: pd.DataFrame, original_feature_names: FeatureTypes):
        """
        Set necessary global variables.
        :param total_num_features:
        :param original_features:
        :param original_feature_names:
        """
        self.__old_features: pd.DataFrame = original_features.copy()
        self.__index: pd.Index = original_features.index.copy()
        self.__total_num_features: int = total_num_features
        self.__level: int = 0
        self.__level_to_features: Dict[int, FeatureTypes] = dict()
        self.__level_to_features[0] = original_feature_names
        self.__new_features = {
            FeatureDataTypes.Numerical: pd.DataFrame(index=self.__index),
            FeatureDataTypes.Categorical: pd.DataFrame(index=self.__index),
            FeatureDataTypes.Boolean: pd.DataFrame(index=self.__index),
            FeatureDataTypes.Date: pd.DataFrame(index=self.__index)
        }
        return

    def set_new_features(self) -> None:
        """

        :return:
        """
        self.__new_features = {
            FeatureDataTypes.Numerical: pd.DataFrame(index=self.__index),
            FeatureDataTypes.Categorical: pd.DataFrame(index=self.__index),
            FeatureDataTypes.Boolean: pd.DataFrame(index=self.__index),
            FeatureDataTypes.Date: pd.DataFrame(index=self.__index)
        }
        return

    def __num_features_helper(self) -> int:
        """

        :return:
        """
        new_features_dfs = [self.__new_features[dtype] for dtype in self.__new_features.keys()]
        new_feature_lens = [nf.shape[1] for nf in new_features_dfs]
        return self.__old_features.shape[1] + sum(new_feature_lens)

    def completed(self) -> bool:
        num_new_features = self.__num_features_helper()
        if num_new_features >= self.__total_num_features:
            return True
        else:
            return False

    def __update_level_to_features(self) -> None:
        """"""
        if self.__level + 1 not in self.__level_to_features.keys():
            self.__level_to_features[self.__level + 1] = FeatureTypes(
                numerical=list(self.__new_features[FeatureDataTypes.Numerical].columns),
                categorical=list(self.__new_features[FeatureDataTypes.Categorical].columns),
                boolean=list(self.__new_features[FeatureDataTypes.Boolean].columns),
                date=list(self.__new_features[FeatureDataTypes.Date].columns))
            pass
        else:
            raise Exception(f'Unexpected behavior in "__update_level_to_features()"')

    def get_new_features(self) -> pd.DataFrame:
        return pd.concat([self.__new_features[key] for key in self.__new_features.keys()], axis=1)

    def concat_new_features(self) -> pd.DataFrame:
        """
        :return: pandas.DataFrame of all current features, including self.__old_features and self.__new_features
        """
        return pd.concat([self.__old_features, self.get_new_features()], axis=1)

    def reset(self) -> None:
        """
        Prepares the class to take on another level of features
        :return:
        """
        self.__old_features = self.concat_new_features()
        self.__update_level_to_features()
        self.set_new_features()
        self.__level += 1
        return

    def add_feature(self, feature, feature_data_type, new_feature_name) -> None:
        self.__new_features[feature_data_type][new_feature_name] = feature

    def get_this_level_feat_names(self) -> FeatureTypes:
        return self.__level_to_features[self.__level]

    def get_col_names(self, feature_data_type: str) -> List[str]:
        """
        summary
            Gets names of columns of the specified data type
        parameters
            feature_data_type: string (one of the FeatureDataTypes instance variables)
        returns
            list of strings with the names of the columns of the specified data type
        """
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
            all_col_names += col_name_list
        return all_col_names

    def get_all_col_names(self):
        """
        :return: dictionary column_data_ype : list<string>, where the
                 strings are the names of the columns
        """
        col_data_types = FeatureDataTypes.get_all_feature_data_types()
        return {col_data_type: self.get_col_names(col_data_type) for col_data_type in col_data_types}

    def get_level_to_feat_names(self) -> Dict[int, FeatureTypes]:
        return self.__level_to_features


class FeatureOperations:
    """

    """

    def __init__(self, max_proportion_null: float, handling_method: str):
        self.max_proportion_null = max_proportion_null
        self.handling_method = handling_method
        return

    def try_add_new_categorical_summary_stat_feat_op_feature(self, groups, all_features: AllFeatures, operation: str,
                                                             cat_feat_name: str, num_feat_name: str,
                                                             check_feature_similarity: bool) -> None:
        """
        :param groups:
        :param all_features:
        :param operation:
        :param cat_feat_name:
        :param num_feat_name:
        :param check_feature_similarity:
        :return:
        """
        new_feature = groups[num_feat_name].transform(operation)
        self.__handle_feature(new_feature, all_features, FeatureDataTypes.Numerical, [num_feat_name, cat_feat_name],
                              operation, check_feature_similarity=check_feature_similarity)
        return

    def try_add_new_single_num_feat_op_feature(self, df: pd.DataFrame, all_features: AllFeatures, single_feat_op: str,
                                               num_feat_name: str, check_feature_similarity: bool) -> None:
        """
        :param df: dataframe where each column is a feature, each row is an example
        :param all_features:
        :param single_feat_op: indicates the type of single feature operation to perform
        :param num_feat_name: name of the column that has the numerical feature to be transformed
        :param check_feature_similarity: flags whether or not to perform the (expensive) action of checking whether
                                         a newly-created synthetic feature is too similar to already existing features.
        :return: None; this method edits new_features in-place (to avoid cost of copying variables)
        """
        new_feature = FeatureOperations._apply_single_num_feat_operation(df[num_feat_name], single_feat_op)
        self.__handle_feature(new_feature, all_features, FeatureDataTypes.Numerical, [num_feat_name], single_feat_op,
                              check_feature_similarity=check_feature_similarity)
        return

    def try_add_new_dual_num_feat_op_feature(self, df, all_features, operation, num_feat_name1, num_feat_name2,
                                             check_feature_similarity: bool):
        """

        :param df:
        :param all_features:
        :param operation:
        :param num_feat_name1:
        :param num_feat_name2:
        :param check_feature_similarity:
        :return:
        """
        new_feature = FeatureOperations._apply_two_num_feat_operation(df[num_feat_name1], df[num_feat_name2], operation)
        self.__handle_feature(new_feature, all_features, FeatureDataTypes.Numerical, [num_feat_name1, num_feat_name2],
                              operation, check_feature_similarity=check_feature_similarity)
        return

    def try_add_new_dual_bool_feat_op_feature(self, df: pd.DataFrame, all_features, operation: str,
                                              bool_feat_name1: str,
                                              bool_feat_name2: str, check_feature_similarity: bool):
        """

        :param df:
        :param all_features:
        :param operation:
        :param bool_feat_name1:
        :param bool_feat_name2:
        :param check_feature_similarity:
        :return:
        """
        new_feature = FeatureOperations._apply_two_bool_feat_operation(df[bool_feat_name1], df[bool_feat_name2],
                                                                       operation)
        self.__handle_feature(new_feature, all_features, FeatureDataTypes.Boolean, [bool_feat_name1, bool_feat_name2],
                              operation, check_feature_similarity=check_feature_similarity)
        return

    def __get_feature_nullity_state(self, feature: pd.Series):
        """
        summary
            self explanatory
        parameters
            feature: pandas.Series
        returns
            the state of the feature (one of the class variables)
        """
        if not isinstance(feature, pd.Series):
            raise _IncorrectTypeException('feature', type(pd.Series()), type(feature))

        num_null = feature.isnull().sum()
        prop_null = num_null / feature.size
        if prop_null > self.max_proportion_null:
            return _FeatureNullityStates.Bad
        elif prop_null > 0:
            return _FeatureNullityStates.Ok
        else:
            return _FeatureNullityStates.Good

    @staticmethod
    def __feature_is_too_uniform(feature: pd.Series, max_prop_same: float = 1.0):
        """
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
        """
        try:
            feature_first_el = feature.values[0]
            same_vals = feature.values == feature_first_el
            num_same_vals = np.sum(same_vals)
            total_num_vals = feature.size
            if num_same_vals / total_num_vals >= max_prop_same:
                return True
            else:
                return False
        except Exception:
            print('An error occurred in __feature_is_uniform(), and it\'s unclear how to handle it.\n'
                  + 'Erring on the side of caution and returning True.')
            return True

    @staticmethod
    def __new_feature_is_too_similar_to_existing_features(new_feature: pd.Series, existing_features: pd.DataFrame,
                                                          max_prop_same: float = 1.0):
        """
        :param new_feature: pandas.Series
        :param existing_features: pandas.DataFrame
        :param max_prop_same: the maximum proportion of elements that can be shared between
                              the new_feature and any given column in existing_features; defaults
                              to 1.0, meaning it will only return True if the entire feature is a duplicate
        :return: bool that is True if the given feature is already a column in
                 the DataFrame (disregarding column names), and False otherwise
        """
        existing_feature_vals = existing_features.values
        num_rows = new_feature.shape[0]
        new_feature_vals = np.reshape(new_feature.values, (num_rows, 1))
        column_equals_feature = existing_feature_vals == new_feature_vals
        num_same_each_column = np.sum(column_equals_feature, axis=0)
        prop_same_each_column = num_same_each_column / num_rows
        cols_entirely_same = (prop_same_each_column == 1)
        if np.sum(cols_entirely_same) > 1:
            raise Exception('Previously there was a feature added that should not have been added (or perhaps some of '
                            + 'the input features were duplicates, which is also no-bueno).')
        cols_too_similar = (prop_same_each_column >= max_prop_same)
        if np.sum(cols_too_similar) >= 1:
            return True
        else:
            return False

    def __is_redundant_feature(self, all_features: AllFeatures, new_feature: pd.Series,
                               check_feature_similarity: bool):
        """
        summary:
            Checks if new_feature is a duplicate of an already existing
            feature, or if new_feature gives no information (for example,
            new_feature is always the same value).
        :param all_features: AllFeatures object
        :param new_feature: pandas.Series
        :return: bool False if feature is too uniform or too similar to other existing features, else False
        """

        existing_features = all_features.concat_new_features()

        if FeatureOperations.__feature_is_too_uniform(new_feature):
            return True
        if check_feature_similarity \
              and self.__new_feature_is_too_similar_to_existing_features(new_feature, existing_features):
            return True
        else:
            return False

    def __handle_feature(self, feature: pd.Series, old_features: AllFeatures, feat_data_type: str, old_feat_names:
                        List[str], operation: str, check_feature_similarity: bool, verbose: bool = False) -> None:
        """
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
        """
        if not isinstance(feature, pd.Series):
            raise _IncorrectTypeException('feature', pd.Series, type(feature))

        # Check for nullity (is the feature too empty?)
        feature_nullity_state = self.__get_feature_nullity_state(feature)
        if feature_nullity_state == _FeatureNullityStates.Good:
            pass  # do nothing, because the feature is already perfect
        elif feature_nullity_state == _FeatureNullityStates.Ok:
            if self.handling_method == _FeatureHandlingMethods.ImputeMedian:
                feature = feature.fillna(feature.median())
            elif self.handling_method == _FeatureHandlingMethods.ImputeMean:
                feature = feature.fillna(feature.mean())
            elif self.handling_method == _FeatureHandlingMethods.Zero:
                feature = feature.fillna(0)
            elif self.handling_method == _FeatureHandlingMethods.Remove:
                return  # don't add feature
            else:
                raise Exception('Invalid handling method; check initialization of this instance'
                                + ' of FeatureOperations; it may help to use a FeatureHandlingMethods'
                                + 'class variable, to avoid typographic mistakes.')
        elif feature_nullity_state == _FeatureNullityStates.Bad:
            if verbose:
                print('from FeatureOperations.handle_features: A feature was labeled as "tooManyNullValues", or'
                      + 'FeatureStates.Bad. Returning None')
            return
        else:
            raise Exception("Invalid Feature State")

        if not self.__is_redundant_feature(old_features, feature, check_feature_similarity=check_feature_similarity):
            # the feature has been handled and is in OK condition to add to new_features
            new_feature_name = FeatureName(feat_data_type, old_feat_names, operation).serialize_feature_name()
            old_features.add_feature(feature, feat_data_type, new_feature_name)
        return

    @staticmethod
    def _apply_single_num_feat_operation(feature: pd.Series, operation_str: str):
        operation_str_to_builtin_operation_funcs = {
            'exp': np.exp,
            'log': np.log,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'sinh': np.sinh,
            'cosh': np.cosh,
            'tanh': np.tanh
        }

        if operation_str in operation_str_to_builtin_operation_funcs.keys():
            operation_func = operation_str_to_builtin_operation_funcs[operation_str]
            return feature.apply(operation_func)  # not safe, because may cause problems if out of range
        else:
            if operation_str == _SingleFeatureOperations.Relu:
                return feature.apply(lambda x: 0 if x <= 0 else x)
            elif operation_str == _SingleFeatureOperations.Sigmoid:
                return feature.apply(lambda x: 1 / (1 + math.exp(-x)))
            elif operation_str == _SingleFeatureOperations.Square:
                return feature.apply(lambda x: x ** 2)
            elif operation_str == _SingleFeatureOperations.Cube:
                return feature.apply(lambda x: x ** 3)
            else:
                raise Exception('"operation_str" not recognized.')

    @staticmethod
    def _apply_two_num_feat_operation(feat1: pd.Series, feat2: pd.Series, operation_str: str):
        if operation_str == _TwoFeatureOperations.Add:
            return feat1 + feat2
        elif operation_str == _TwoFeatureOperations.Subtract:
            return feat1 - feat2
        elif operation_str == _TwoFeatureOperations.Multiply:
            return feat1 * feat2

    @staticmethod
    def _apply_two_bool_feat_operation(feature1: pd.Series, feature2: pd.Series, operation_str: str):
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

        if operation_str == _BooleanOperations.Xor:
            func = xor_func
        elif operation_str == _BooleanOperations.And:
            func = and_func
        elif operation_str == _BooleanOperations.Or:
            func = or_func
        elif operation_str == _BooleanOperations.Nand:
            func = nand_func

        new_feature = combined.apply(lambda x: func(x), axis=1)
        if not isinstance(new_feature, pd.Series):
            raise _IncorrectTypeException('new_feature', type(pd.Series()), type(new_feature))
        return new_feature


class FeatureSynthesis:
    __DEFAULT_FEATURE_HANDLING_METHOD = _FeatureHandlingMethods.ImputeMean
    __DEFAULT_MAX_PROPORTION_NULL_VALUES = 0.1
    __DEFAULT_CHECK_FEATURE_SIMILARITY = True

    def __init__(self):
        self.__feature_names: Dict[str, List[str]] = dict()  # maps from data type to feature names of that data type
        self.__level_to_feature_names: Dict[int, FeatureTypes] = dict()
        self.__features_from_fit: pd.DataFrame = pd.DataFrame()
        self.__has_been_fit = False
        return

    def fit(self, original_features: pd.DataFrame, original_feature_names: FeatureTypes,
            total_num_features: int,
            max_prop_null: float = __DEFAULT_MAX_PROPORTION_NULL_VALUES,
            null_handling_method: str = __DEFAULT_FEATURE_HANDLING_METHOD,
            check_feature_similarity: bool = __DEFAULT_CHECK_FEATURE_SIMILARITY) -> pd.DataFrame:

        assert isinstance(original_feature_names, FeatureTypes), "Input feature_names is not of type FeatureTypes!"

        def try_finish_helper(all_features: AllFeatures) -> bool:
            if all_features.completed():
                self.__features_from_fit = all_feats.concat_new_features()
                all_features.reset()
                self.__set_feature_names(all_features)
                self.__has_been_fit = True
                return True
            else:
                return False

        feat_ops: FeatureOperations = FeatureOperations(max_prop_null, null_handling_method)
        all_feats: AllFeatures = AllFeatures(total_num_features=total_num_features, original_features=original_features,
                                             original_feature_names=original_feature_names)

        while True:
            feature_names = all_feats.get_this_level_feat_names()

            # Populate with synthetic features related to groups and categorical variables
            for cat_feat_name in feature_names.Categorical:
                category_groups = original_features.groupby(cat_feat_name)
                for num_feat_name in feature_names.Numerical:
                    for summary_stat in _NumericalSummaryStats.get_all_operations():
                        if try_finish_helper(all_feats):
                            return self.__features_from_fit
                        else:
                            feat_ops.try_add_new_categorical_summary_stat_feat_op_feature(category_groups, all_feats,
                                                                                          summary_stat, cat_feat_name,
                                                                                          num_feat_name,
                                                                                          check_feature_similarity)
                # TODO: For each group of categorical features, give its most frequent entity in all other groups

            # Populate with synthetic features related to numerical transformations
            for i, num_feat_name in enumerate(feature_names.Numerical):
                for single_feat_op in _SingleFeatureOperations.get_all_operations():
                    if try_finish_helper(all_feats):
                        return self.__features_from_fit
                    else:
                        feat_ops.try_add_new_single_num_feat_op_feature(original_features, all_feats,
                                                                        single_feat_op, num_feat_name,
                                                                        check_feature_similarity)

                # avoids pairs of features from getting called twice; also avoids pairing with self
                other_num_feature_names = feature_names.Numerical[i + 1:]
                for other_num_feat_name in other_num_feature_names:
                    for dual_feat_op in _TwoFeatureOperations.get_all_operations():
                        if try_finish_helper(all_feats):
                            return self.__features_from_fit
                        else:
                            feat_ops.try_add_new_dual_num_feat_op_feature(original_features, all_feats, dual_feat_op,
                                                                          num_feat_name, other_num_feat_name,
                                                                          check_feature_similarity)

            # Populate with synthetic features related to boolean transformations
            for i, bool_feat_name in enumerate(feature_names.Boolean):
                other_bool_feat_names = feature_names.Boolean[i + 1:]
                for other_bool_feat_name in other_bool_feat_names:
                    for bool_op in _BooleanOperations.get_all_operations():
                        if try_finish_helper(all_feats):
                            return self.__features_from_fit
                        else:
                            feat_ops.try_add_new_dual_bool_feat_op_feature(original_features, all_feats, bool_op,
                                                                           bool_feat_name, other_bool_feat_name,
                                                                           check_feature_similarity)

            # TODO: Create Date and Categorical synthetic features

            original_features = all_feats.get_new_features()
            all_feats.reset()

    def transform(self, original_features: pd.DataFrame) -> pd.DataFrame:
        """
        Converts each row of 'features' to an image.
        :param original_features: features in original format (same format as features given as input to 'fit')
        :return:
        """
        assert self.__has_been_fit, "Model must be fit before being transformed."

        return self.__create_synthetic_features(original_features)

    def __create_synthetic_features(self, original_features):
        """
        Takes in current features, and tries to synthesize all features that were created
        in fitting, without any leakage of information.
        :param original_features:
        :return:
        """

        # * Make a dataframe copying original features.
        # * Get the features at each level
        #   * For each level of fit_feature, recreate it using the
        #     the features in original_features_copy

        features = original_features.copy()

        if len(self.__level_to_feature_names.keys()) != 0:
            synthetic_levels: List[int] = list(self.__level_to_feature_names.keys())[1:]
            for level in synthetic_levels:
                feature_types: Dict[str, List[str]] = self.__level_to_feature_names[level].to_dict()
                for feature_names in feature_types.values():
                    for new_feature_name in feature_names:
                        self.__create_synthetic_feature(features, new_feature_name)

        return features

    def __create_synthetic_feature(self, existing_features: pd.DataFrame, new_feature_name: str) -> None:
        """
        Creates a new synthetic feature and adds it in place to existing_features.
        :param existing_features:
        :param new_feature_name:
        :return:
        """
        assert new_feature_name not in existing_features.columns.values, "Feature already exists!"

        new_feature_obj: FeatureName = FeatureName.deserialize_feature_name(new_feature_name)
        operation = new_feature_obj.operation
        if operation in _SingleFeatureOperations.get_all_operations():
            old_feat_name = new_feature_obj.old_feat_names[0]
            old_feat = existing_features[old_feat_name]
            existing_features[new_feature_name] = \
                FeatureOperations._apply_single_num_feat_operation(old_feat, operation)
        elif operation in _TwoFeatureOperations.get_all_operations():
            feat_name1, feat_name2 = new_feature_obj.old_feat_names
            feat1, feat2 = existing_features[feat_name1], existing_features[feat_name2]
            existing_features[new_feature_name] = \
                FeatureOperations._apply_two_num_feat_operation(feat1, feat2, operation)
        elif operation in _BooleanOperations.get_all_operations():
            feat_name1, feat_name2 = new_feature_obj.old_feat_names
            feat1, feat2 = existing_features[feat_name1], existing_features[feat_name2]
            existing_features[new_feature_name] = \
                FeatureOperations._apply_two_bool_feat_operation(feat1, feat2, operation)
        elif operation in _NumericalSummaryStats.get_all_operations():
            # The first is the numerical variable, then the categorical variable.
            _, cat_feat_name = new_feature_obj.old_feat_names
            categorical_mappings = self.__get_cat_mappings(new_feature_name)
            existing_features[new_feature_name] = existing_features[cat_feat_name].map(categorical_mappings)
        else:
            raise Exception(f'Invalid feature operation: "{operation}".')

        return

    def __get_cat_mappings(self, new_feature_name: str) -> Dict:
        """
        Given the categorical and numerical feature names, this method will look into the
        features that were used when fitting, and create a mapping from each unique categorical variable
        to its corresponding numerical summary statistic.
        :return:
        """
        new_feat_obj = FeatureName.deserialize_feature_name(new_feature_name)
        num_feat_name, cat_feat_name = new_feat_obj.old_feat_names
        cat_col, summary_stat_col = self.__features_from_fit[cat_feat_name], self.__features_from_fit[new_feature_name]
        unique_cats = cat_col.unique()
        cat_to_summary_stat = {}
        for cat in unique_cats:
            is_this_cat: pd.Series = (cat_col == cat)  # gets indices where categorical var matches
            this_summary_stat_col = summary_stat_col[is_this_cat]
            if this_summary_stat_col.size == 0:
                print("Caution: did not see this category when fitting FeatureSynthesis instance. Using the mean"
                      "of the column in place of the true categorical variable.")
                cat_to_summary_stat[cat] = summary_stat_col.mean()
            else:
                # Although there is no assertion here, it should be the case that
                # all elements in this_summary_stat_col are identical (100% uniform).
                # If this were validated every time, it may slow down the program.
                cat_to_summary_stat[cat] = this_summary_stat_col.iloc[0]

        return cat_to_summary_stat

    def __set_feature_names(self, all_features: AllFeatures):
        """

        :param all_features:
        :return:
        """
        self.__feature_names = all_features.get_all_col_names()
        self.__level_to_feature_names = all_features.get_level_to_feat_names()
        return

    def get_feature_names(self, data_type):
        """
        :param data_type: string denoting data type; use FeatureDataTypes class variables for more secure code
        :return: list<string> with feature names
        """
        return self.__feature_names[data_type]


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

    example_df, example_feature_names = make_examples_helper(100)

    import time

    t0 = time.time()
    fs = FeatureSynthesis()
    new_feats = fs.fit(
        original_features=example_df, original_feature_names=example_feature_names, total_num_features=100,
        check_feature_similarity=True)
    t1 = time.time()
    print(new_feats.head())
    print(f'Took {t1 - t0} seconds to finish.')

    new_examples_df, _ = make_examples_helper(10)
    new_stuff = fs.transform(new_examples_df)
    pass