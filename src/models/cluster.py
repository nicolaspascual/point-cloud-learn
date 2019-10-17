import pandas as pd
import re
from io import StringIO


class Cluster(object):
    feature_names = [
        'cluster_name', 'x', 'y', 'z', 'n_points', 'n_order', 'volume',
        'positive_volume', 'negative_volume', 'area', 'classification',
        'confidence', 'class_1_mean', 'class_1_sigma', 'class_2_mean', 'class_2_sigma',
        'class_3_mean', 'class_3_sigma', 'class_4_mean', 'class_4_sigma',
        'n_orientations_origin', 'n_orientations_destination', 'texture_code_origin',
        'red_mean_origin', 'red_sigma_origin', 'green_mean_origin', 'green_sigma_origin',
        'blue_mean_origin', 'blue_sigma_origin', 'intensity_mean_origin',
        'intensity_sigma_origin', 'texture_code_destination', 'red_mean_destination',
        'red_sigma_destination', 'green_mean_destination', 'green_sigma_destination',
        'blue_mean_destination', 'blue_sigma_destination', 'intensity_mean_destination',
        'intensity_sigma_destination', 'correlation_index', 'orientation_mean_origin',
        'slope_mean_origin', 'orientation_mean_destination', 'slope_mean_destination',
        'coplanararity_index_mean_origin', 'coplanararity_index_sigma_origin',
        'colinearity_index_mean_origin', 'colinearity_index_sigma_origin',
        'coplanararity_index_mean_destination', 'coplanararity_index_sigma_destination',
        'colinearity_index_mean_destination', 'colinearity_index_sigma_destination',
        'angles_mean', 'angles_sigma', 
        'file_origin', 'file_destination'
    ]

    @classmethod
    def from_string(cls, string_representation):
        values = re.compile(r'\s+').split(string_representation.strip())[1:]
        return Cluster(*values)

    def __init__(self, *args):
        self.feature_names = Cluster.feature_names[:]
        self.values = args
        self._remove_invalid_texture_cols()
        assert(len(self.values) == len(self.feature_names))
    
    def _remove_invalid_texture_cols(self):
        self.remove_invalid_texture_cols('origin', self.values[22])
        destination_index = 31
        if self.values[22] == '0': destination_index -= 8
        elif self.values[22] == '1': destination_index -= 6
        elif self.values[22] == '2': destination_index -= 2
        self.remove_invalid_texture_cols('destination', self.values[destination_index])

    def remove_invalid_texture_cols(self, suffix, index):
        to_remove = []
        if index in ['1', '0']:
            to_remove += [
                f'red_mean_{suffix}', f'red_sigma_{suffix}',
                f'green_mean_{suffix}', f'green_sigma_{suffix}',
                f'blue_mean_{suffix}', f'blue_sigma_{suffix}'
            ]
        if index in ['2', '0']:
            to_remove += [
                f'intensity_mean_{suffix}', f'intensity_sigma_{suffix}'
            ]
        [self.feature_names.remove(el) for el in to_remove]

    def tabular_representation(self):
        tab_string_rep = ''
        current_value = 0
        for feat_name in Cluster.feature_names:
            if feat_name in self.feature_names:
                tab_string_rep += f'{self.values[current_value]},'
                current_value += 1
            else:
                tab_string_rep += f'None,'
        return tab_string_rep

    def to_series(self):
        return pd.read_csv(
            StringIO(self.tabular_representation()), index_col=False,
            names=Cluster.feature_names
        ).set_index('cluster_name').iloc[0]
