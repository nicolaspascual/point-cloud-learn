import pandas as pd
import re
from io import StringIO


class RepresentativePoint(object):
    feature_names = [
        'texture_code_origin', 'texture_code_destination', 'x_origin', 'y_origin',
        'z_origin', 'vector_i_origin', 'vetor_j_origin', 'vector_k_origin',
        'vector_origin_origin', 'vector_slope_origin',
        'interpolation_quality_index_linearity_origin',
        'interpolation_quality_index_planarity_origin',
        'red_origin', 'green_origin', 'blue_origin', 'intensity_origin',
        'index_origin', 'index_destination', 'x_destination', 'y_destination',
        'z_destination', 'vector_i_destination', 'vetor_j_destination',
        'vector_k_destination', 'vector_origin_destination',
        'vector_slope_destination', 'interpolation_quality_index_linearity_destination',
        'interpolation_quality_index_planarity_destination', 'red_destination',
        'green_destination', 'blue_destination', 'intensity_destination',
        'classification', 'percentage_class_0', 'percentage_class_1', 'percentage_class_2',
        'max_diff', 'average_diff', 'std_diff', 'n_points', 'min_diff', 'vector_distance',
        'horizontal_distance', 'total_distance', 'n_points_voxel', 'angle', 'angle_2n',
        'min_distance', 'mean_dsitance', 'max_distance', 'std_distance', 'n_dist'
    ]

    @classmethod
    def from_string(cls, string_representation):
        values = re.compile(r'\s+').split(string_representation.strip())
        return RepresentativePoint(*values)

    def __init__(self, *args):
        self.feature_names = RepresentativePoint.feature_names[:]
        self.values = args
        self.remove_invalid_texture_cols('origin', self.values[0])
        self.remove_invalid_texture_cols('destination', self.values[1])

    def remove_invalid_texture_cols(self, suffix, index):
        to_remove = []
        if index in ['1', '0']:
            to_remove += [
                f'red_{suffix}', f'green_{suffix}', f'blue_{suffix}'
            ]
        if index in ['2', '0']:
            to_remove += [
                f'intensity_{suffix}'
            ]
        [self.feature_names.remove(el) for el in to_remove]

    def tabular_representation(self):
        tab_string_rep = ''
        current_value = 0
        for feat_name in RepresentativePoint.feature_names:
            if feat_name in self.feature_names:
                tab_string_rep += f'{self.values[current_value]},'
                current_value += 1
            else:
                tab_string_rep += f'None,'
        return tab_string_rep

    def to_series(self):
        return pd.read_csv(
            StringIO(self.tabular_representation()), index_col=False,
            names=RepresentativePoint.feature_names
        ).iloc[0]
