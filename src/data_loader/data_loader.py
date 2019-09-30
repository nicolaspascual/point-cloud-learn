import re
import pandas as pd
from io import StringIO
from src.models import Cluster, RepresentativePoint

def load_data(file_name):
    with open(file_name, 'r') as file:
        clusters = []
        representative_points = []
        for line in file.readlines()[::-1]:
            if re.compile(r'^#cluster: \d+').match(line):
                clusters.append(
                    Cluster.from_string(line)
                )
            else:
                representative_points.append(
                    RepresentativePoint.from_string(line)
                )
    cluster_str = StringIO(
        '\n'.join([cluster.tabular_representation() for cluster in clusters])
    )
    repre_str = StringIO(
        '\n'.join([point.tabular_representation() for point in representative_points])
    )
    return pd.read_csv(cluster_str, index_col=False, names=Cluster.feature_names)\
                .set_index('cluster_name'),\
        pd.read_csv(repre_str, names=RepresentativePoint.feature_names, index_col=False)
