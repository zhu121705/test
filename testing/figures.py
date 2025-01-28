from matplotlib import pyplot as plt
import pandas as pd
from query.utils import read_yaml
from query import query_api
from collections import OrderedDict
import os

model_size_df = pd.read_csv('csv/model_size.csv')
config = read_yaml('configurations/example_configuration.yaml')['configuration']

output_dir = 'figures/'
os.makedirs(output_dir, exist_ok=True)

def plot(label, year_dict: OrderedDict):
    plt.figure(figsize=(20, 10))
    # Convert keys to floats and plot both line and scatter points
    years = [float(year) for year in year_dict.keys()]
    values = list(year_dict.values())
    print(years, values)
    plt.plot(years, values, label=label, marker='o', 
             markersize=8, linestyle='-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Parameters (B)', fontsize=12)
    plt.title(f'{label} Model Size Trend', fontsize=14)
    plt.xlim(2016, 2025)  # Set x-axis range
    plt.xticks(range(2016, 2026))  # Set x-axis ticks for each year
    plt.legend(fontsize=10)
    plt.savefig(os.path.join(output_dir, f'{label}_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

def query_function(plot_dict: OrderedDict, function):
    query_obj = query_api.Query('configurations/example_configuration.yaml', verbose = False, function=function)

    year_query_function = 'year_query_' + function

    queries_dict = OrderedDict({
        'queries': OrderedDict({})
    })

    for year_range in config['year_ranges']['model_size']:
        year_range = (year_range[0], year_range[1])
        average_year = sum(year_range) // 2
        queries_dict['queries'][average_year] = OrderedDict({
            'year': average_year,
        })
        plot_dict[year_query_function][str(sum(year_range) / 2)] = None

    queries = query_obj.queries(queries_dict = queries_dict, log = False)

    # Replace the single for loop with a zip iteration
    for year_key in plot_dict[year_query_function].keys():
        plot_dict[year_query_function][year_key] = queries['queries'][int(float(year_key))]['parameters']
    return plot_dict

plot_dict = OrderedDict({
    'min_year': OrderedDict(),
    'max_year': OrderedDict(),
    'average_year': OrderedDict(),
    'year_query_mean': OrderedDict(),
    'year_query_median': OrderedDict(),
    'year_query_geomean': OrderedDict()
})

for index, row in model_size_df.iterrows():
    year = row['Year']
    params = row['# Params (B)']
    if year not in plot_dict['min_year']:
        plot_dict['min_year'][year] = params
        plot_dict['max_year'][year] = params
        plot_dict['average_year'][year] = [params]
    else:
        plot_dict['min_year'][year] = min(plot_dict['min_year'][year], params)
        plot_dict['max_year'][year] = max(plot_dict['max_year'][year], params)
        plot_dict['average_year'][year].append(params)

for year in plot_dict['average_year']:
    plot_dict['average_year'][year] = sum(plot_dict['average_year'][year]) / len(plot_dict['average_year'][year])

plot_dict = query_function(plot_dict, 'mean')
plot_dict = query_function(plot_dict, 'median')
plot_dict = query_function(plot_dict, 'geomean')

for key, value in plot_dict.items():
    plot(key, value)
