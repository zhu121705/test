from query.query_api import Query

config_dir = 'checkpoints/'
config_file = 'checkpoint_0.pkl'
config_path = config_dir + config_file

query_dir = 'example_queries/'
query_file = 'example_queries_1.yaml'
query_path = query_dir + query_file

queries = Query(config_path, verbose = True)

output_dict = queries.queries(query_path=query_path, log = True)

print(output_dict)

queries.save_pkl()