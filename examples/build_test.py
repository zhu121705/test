from query.utils import build_test
from query.query_api import Query

# test dict keys must be the same as the keys used to query the api
# if you dont want any of the keys, do not include them. If you want the case where the key is not used, include None in the list
test_dict = {
    'year': [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'phase': ['inference', 'training', None],
    'numerical_format': ['32-16', '16-8', '16-4', None],
    'parallel_strategy': ['pipeline', 'tensor', 'automated search', None],
    'optimizer': ['sgd', 'rmsprop', 'adam', None],
    'chip_type': ['gpu', 'acc', 'wse', None]
}

build_test(test_dict, path='example_queries/test.yaml')

# uncomment the following lines to run the example test, which runs every combination of the test_dict

# config_dir = 'configurations/'
# config_file = 'example_configuration.yaml'
# config_path = config_dir + config_file

# query_dir = 'example_queries/'
# query_file = 'test.yaml'
# query_path = query_dir + query_file

# queries = Query(config_path, verbose = True)

# output_dict = queries.queries(query_path = query_path, log = True)