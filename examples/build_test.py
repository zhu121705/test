from query.utils import build_test

# test dict keys must be the same as the keys used to query the api
# if you dont want any of the keys, do not include them. If you want the case where the key is not used, include None in the list
test_dict = {
    'year': [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'phase': ['inference', 'training'],
    'numerical_format': ['32-16', '16-8', '16-4'],
    'parallel_strategy': ['pipeline', 'tensor', 'automated search'],
    'optimizer': ['sgd', 'rmsprop', 'adam'],
    'chip_type': ['gpu', 'acc', 'wse']
}

build_test(test_dict, path='example_queries/test.yaml')