from query.query_api import Query

config_dir = 'configurations/'
config_file = 'example_configuration.yaml'
config_path = config_dir + config_file

query_dir = 'example_queries/'
query_file = 'example_query_1.yaml'
query_path = query_dir + query_file

# changes interanl average function from mean to median
queries = Query(config_path, verbose = True, function = 'median')

output_dict = queries.query(query_path = query_path, log = True, numerical_format_scaling = False)
# additionally, you can also turn off internal runtime scaling variables, theses options are 'numerical_format_scaling', 'parallel_strategy_scaling', and 'hardware_configuration_scaling'
# ex. output_dict = queries.query(query_path = query_path, log = True, function = 'median', numerical_format_scaling = False, parallel_strategy_scaling = False, hardware_configuration_scaling = False)
#                                                                       ------------------------------------------------------------------------------------------------------------
# In the example above, the output_dict will still return the numerical format, but it will not scale the runtime based on the numerical format speedup. To see this, compare the output dict 
# with numerical_format_scaling = True and numerical_format_scaling = False\

print(output_dict)

output_dict = queries.query(query_path = query_path, log = True,) # Defaults to true

print(output_dict)

queries.save_pkl()