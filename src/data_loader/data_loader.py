import re
from collections import namedtuple

Cluster = namedtuple('Cluster', [
        'cluster_id', 'x', 'y', 'z', 'n_points', 'n_order', 'volume', 'positive_volume',
        'negative_volume', 'area_2d', 'klass', 'trust_level', 'p_all_mean', 'p_all_sigma',
        'p_1_mean', 'p_1_sigma', 'p_0_mean', 'p_0_sigma', 'p_2_mean', 'p_2_sigma',
        'n_clusters_origin', 'n_clusters_destination', 'color_origin_index', 'red_origin_mean',
        'red_origin_sigma', 'green_origin_mean', 'green_origin_sigma', 'blue_origin_mean',
        'blue_origin_sigma', 'intensity_origin_mean', 'intensity_origin_sigma',
        'color_reference_index', 'red_reference_mean', 'red_reference_sigma',
        'green_reference_mean', 'green_reference_sigma', 'blue_reference_mean',
        'blue_reference_sigma', 'intensity_destination_mean', 'intensity_destination_sigma',
        'origin_file', 'destination_file'
    ])
ClusterRepresentativePoint = namedtuple('ClusterRepresentativePoint', [])

def load_data(file_name):
    with open(file_name, 'r') as file:
        clusters = []
        representative_points = [[]]
        for line in file.readlines():
            if re.compile(r'^#cluster: \d+').match(line):
                clusters.append(
                    parse_cluster(line)
                )
                representative_points.append([])
            else:
                representative_points[-1].append(
                    parse_point(line)
                )
        representative_points.pop()
    return clusters, representative_points


def parse_cluster(line: str):
    line = line.replace('#cluster:', '')
    fields = re.compile(r'\s+').split(line.strip())
    return Cluster(*fields)

def parse_point(line):
    return re.compile(r'\s+').split(line.strip())