import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
from query.utils import read_yaml
import numpy as np

def plot_trendline(data_dict: OrderedDict, label: str):
    plt.figure(figsize=(20, 10))
    
    # Define the correct order for formats and colors
    format_order = ['32-16', '16-8', '16-4']
    colors = {
        'mean': '#3366cc',    # blue
        'median': '#dc3912',  # red
        'std': '#109618'      # green
    }
    
    # Create mapping based on predefined order
    format_to_pos = {fmt: i for i, fmt in enumerate(format_order)}
    
    # Plot each metric (mean, median, std)
    for metric, values in data_dict.items():
        x_points = [format_to_pos[fmt] for fmt in format_order if fmt in values]
        y_points = [values[fmt] for fmt in format_order if fmt in values]
        
        plt.plot(x_points, y_points,
                label=f'{metric.capitalize()}',
                linestyle='-', linewidth=3,
                marker='o', markersize=8,
                c=colors[metric])
    
    # Use predefined order for x-axis labels
    plt.xticks(range(len(format_order)), format_order, rotation=45)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Format Type', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(f'{label} Metrics', fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join('figures', f'{label}_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_grouped_metrics(data_dict: OrderedDict, metric_type: str, label: str, format_order: int = 0):
    plt.figure(figsize=(20, 10))
    
    format_order = ['32-16', '16-8', '16-4'] if format_order == 0 else ['pipeline', 'tensor', 'automated search']
    colors = {
        'geomean': '#66a3ff',    # lighter blue
        'mean': '#ff6666',       # lighter red
        'median': '#66cc66'      # lighter green
    }
    
    format_to_pos = {fmt: i for i, fmt in enumerate(format_order)}
    
    # Plot the specified metric (mean or std) for each average type
    for avg_type, values in data_dict.items():
        x_points = [format_to_pos[fmt] for fmt in format_order if fmt in values[metric_type]]
        y_points = [values[metric_type][fmt] for fmt in format_order if fmt in values[metric_type]]
        
        plt.plot(x_points, y_points,
                label=f'{avg_type.capitalize()}',
                linestyle='-', linewidth=3,
                marker='o', markersize=8,
                c=colors[avg_type])
    
    plt.xticks(range(len(format_order)), format_order, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Format Type', fontsize=12)
    plt.ylabel(f'{metric_type.capitalize()} Value', fontsize=12)
    plt.title(f'{label} {metric_type.capitalize()} Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join('figures', f'{label}_{metric_type}_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()



def plot(scatter: list[tuple], line: list[tuple], label: str, year_bool: bool = False, type: str = ''):
    plt.figure(figsize=(20, 10))

    scatter_y = [value for value, _ in scatter]
    line_y = [value for value, _ in line]
    if year_bool:
        scatter_x = [int(x) for _, x in scatter]
        line_x = [float(x) for _, x in line]
    else:
        format_order = ['32-16', '16-8', '16-4'] if type == 'nf' else ['pipeline', 'tensor', 'automated search']
        scatter_x = [format_order.index(x) for _, x in scatter]
        line_x = [format_order.index(x) for _, x in line]

    plt.scatter(scatter_x, scatter_y, 
                label=f'{label} Data Points', 
                marker='o', s=64, 
                c='#ff9900', alpha=0.75)
    sorted_line = sorted(zip(line_x, line_y))

    line_dict = OrderedDict({})
    for x, y in sorted_line:
        if y not in line_dict:
            line_dict[y] = []
        line_dict[y].append(x)
    grouped_data = line_dict
    for key, value in line_dict.items():
        line_dict[key] = np.mean(value)
    sorted_x = [x for x in line_dict.values()]
    sorted_y = [y for y in line_dict.keys()]

    config = read_yaml('configurations/example_configuration.yaml')['configuration']
    year_ranges = config['year_ranges']['numerical_format'] if type == 'nf' else config['year_ranges']['parallel_strategy']

    scatter_dict = OrderedDict()
    for x, y in zip(scatter_x, scatter_y):
        if x not in scatter_dict:
            scatter_dict[x] = []
        scatter_dict[x].append(y)

    # Prepare boxplot data
    box_data = [scatter_dict[x] for x in sorted(scatter_dict.keys())]
    box_pos = sorted(scatter_dict.keys())

    box_pos_new = [np.mean(year_range) for year_range in year_ranges] if year_bool else box_pos
    box_data_new = [[] for _ in range(len(year_ranges))] if year_bool else box_data
    
    if year_bool:
        for x, y in zip(box_pos, box_data):
            for i, year_range in enumerate(year_ranges):
                if float(year_range[0]) <= float(x) <= float(year_range[1]):
                    box_data_new[i] = y

    plt.boxplot(box_data_new, positions=box_pos_new, 
                    patch_artist=True,
                    medianprops=dict(color="red"),
                    boxprops=dict(facecolor='#3366cc', alpha=0.5))

    plt.plot(sorted_x, sorted_y, label = f'{label} Trendline', linestyle='-', linewidth=3, c='#3366cc')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    if year_bool:
        plt.xlabel('Year', fontsize=12)
    elif type == 'nf':
        plt.xlabel('Format', fontsize=12)
    else:
        plt.xlabel('Parallel Strategy', fontsize=12)

    plt.ylabel('Speedup (X)', fontsize=12)
    plt.title(f'{label} Validation', fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    type = type + '_year' if year_bool else type
    plt.savefig('figures/' + type + '_png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_sl(scatter: list[tuple], line: list[tuple], label: str, year_bool: bool = False, type: str = ''):
    plt.figure(figsize=(20, 10))
    
    scatter_x = [value for value, _ in scatter]
    line_x = [value for value, _ in line]
    if year_bool:
        scatter_y = [int(x) for _, x in scatter]
        line_y = [int(x) for _, x in line]
    else:
        format_order = ['32-16', '16-8', '16-4'] if type == 'nf' else ['pipeline', 'tensor', 'automated search']
        scatter_y = [format_order.index(x) for _, x in scatter]
        line_y = [format_order.index(x) for _, x in line]
    

    # Create plots
    plt.scatter(scatter_x, scatter_y, 
                label=f'{label} Data Points', 
                marker='o', s=64, 
                c='#ff9900', alpha=0.75)
    
    # Sort line points by x position before plotting
    sorted_line = sorted(zip(line_x, line_y))
    print(sorted_line)
    plt.plot([x for x, _ in sorted_line], [y for _, y in sorted_line],
             label=f'{label} Query', 
             linestyle='-', linewidth=3,
             c='#3366cc')
    
    # Use predefined order for x-axis labels
    if not year_bool:
        plt.xticks(range(len(format_order)), format_order, rotation=45)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Format Type', fontsize=12)
    plt.ylabel('Speedup', fontsize=12)
    plt.title(f'{label} Distribution', fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join('figures', f'{label}_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

def average(year_list, pe_list, dict):
    for year, pe in zip(year_list, pe_list):
        if year not in dict['median']:
            dict['median'][year] = []
            dict['mean'][year] = []
            dict['std'][year] = []
        dict['median'][year].append(pe)
        dict['mean'][year].append(pe)
        dict['std'][year].append(pe)

    for year in dict['median']:
        dict['median'][year] = np.median(dict['median'][year])
        dict['mean'][year] = np.mean(dict['mean'][year])
        dict['std'][year] = np.std(dict['std'][year])

results_dir = 'testing/nf_ps_csv/'
raw_data_dir = 'csv/'

config = read_yaml('configurations/example_configuration.yaml')['configuration']

nf_geomean_df = pd.read_csv(results_dir + 'nf_results_geomean.csv')
nf_mean_df = pd.read_csv(results_dir + 'nf_results_mean.csv')
nf_median_df = pd.read_csv(results_dir + 'nf_results_median.csv')
ps_geomean_df = pd.read_csv(results_dir + 'ps_results_geomean.csv')
ps_mean_df = pd.read_csv(results_dir + 'ps_results_mean.csv')
ps_median_df = pd.read_csv(results_dir + 'ps_results_median.csv')

nf_raw_set = list(zip(nf_geomean_df['nf_speedup'].to_list(), nf_geomean_df['nf_format'].to_list()))
nf_raw_year_set = list(zip(nf_geomean_df['year_speedup'].to_list(), nf_geomean_df['nf_year'].to_list()))
ps_raw_set = list(zip(ps_geomean_df['ps_speedup'].to_list(), ps_geomean_df['ps_strategy'].to_list()))
ps_raw_year_set = list(zip(ps_geomean_df['year_speedup'].to_list(), ps_geomean_df['ps_year'].to_list()))

nf_query_set = set(zip(nf_geomean_df['nf_query_speedup'].to_list(), nf_geomean_df['nf_format'].to_list()))
nf_query_year_set = set(zip(nf_geomean_df['year_query_speedup'].to_list(), nf_geomean_df['nf_year'].to_list()))
ps_query_set = set(zip(ps_geomean_df['ps_query_speedup'].to_list(), ps_geomean_df['ps_strategy'].to_list()))
ps_query_year_set = set(zip(ps_geomean_df['year_query_speedup'].to_list(), ps_geomean_df['ps_year'].to_list()))

plot(nf_raw_set, nf_query_set, 'Numerical Format', year_bool=False, type='nf')
plot(nf_raw_year_set, nf_query_year_set, 'Numerical Format', year_bool=True, type='nf')
plot(ps_raw_set, ps_query_set, 'Parallel Strategy', year_bool=False, type='ps')
plot(ps_raw_year_set, ps_query_year_set, 'Parallel Strategy', year_bool=True, type='ps')

# nf_geomean_pe = nf_geomean_df['nf_pe'].to_list()
# nf_geomean_format = nf_geomean_df['nf_format'].to_list()
# nf_geomean_year = nf_geomean_df['nf_year'].to_list()
# nf_mean_pe = nf_mean_df['nf_pe'].to_list()
# nf_mean_format = nf_mean_df['nf_format'].to_list()
# nf_mean_year = nf_mean_df['nf_year'].to_list()
# nf_median_pe = nf_median_df['nf_pe'].to_list()
# nf_median_format = nf_median_df['nf_format'].to_list()
# nf_median_year = nf_median_df['nf_year'].to_list()
# ps_geomean_pe = ps_geomean_df['ps_pe'].to_list()
# ps_geomean_format = ps_geomean_df['ps_strategy'].to_list()
# ps_geomean_year = ps_geomean_df['ps_year'].to_list()
# ps_mean_pe = ps_mean_df['ps_pe'].to_list()
# ps_mean_format = ps_mean_df['ps_strategy'].to_list()
# ps_mean_year = ps_mean_df['ps_year'].to_list()
# ps_median_pe = ps_median_df['ps_pe'].to_list()
# ps_median_format = ps_median_df['ps_strategy'].to_list()
# ps_median_year = ps_median_df['ps_year'].to_list()

# nf_dict = OrderedDict({
#     'geomean': OrderedDict(),
#     'mean': OrderedDict(),
#     'median': OrderedDict(),
# })
# ps_dict = OrderedDict({
#     'geomean': OrderedDict(),
#     'mean': OrderedDict(),
#     'median': OrderedDict(),
# })

# nf_year_dict = OrderedDict({
#     'geomean': OrderedDict(),
#     'mean': OrderedDict(),
#     'median': OrderedDict(),
# })
# ps_year_dict = OrderedDict({
#     'geomean': OrderedDict(),
#     'mean': OrderedDict(),
#     'median': OrderedDict(),
# })

# for value in nf_dict.values():
#     value['mean'] = OrderedDict()
#     value['median'] = OrderedDict()
#     value['std'] = OrderedDict()

# for value in ps_dict.values():
#     value['mean'] = OrderedDict()
#     value['median'] = OrderedDict()
#     value['std'] = OrderedDict()

# for value in nf_year_dict.values():
#     value['mean'] = OrderedDict()
#     value['median'] = OrderedDict()
#     value['std'] = OrderedDict()

# for value in ps_year_dict.values():
#     value['mean'] = OrderedDict()
#     value['median'] = OrderedDict()
#     value['std'] = OrderedDict()

# average(nf_geomean_format, nf_geomean_pe, nf_dict['geomean'])
# average(nf_mean_format, nf_mean_pe, nf_dict['mean'])
# average(nf_median_format, nf_median_pe, nf_dict['median'])
# average(ps_geomean_format, ps_geomean_pe, ps_dict['geomean'])
# average(ps_mean_format, ps_mean_pe, ps_dict['mean'])
# average(ps_median_format, ps_median_pe, ps_dict['median'])

# average(nf_geomean_year, nf_geomean_pe, nf_dict['geomean'])
# average(nf_mean_year, nf_mean_pe, nf_dict['mean'])
# average(nf_median_year, nf_median_pe, nf_dict['median'])
# average(ps_geomean_year, ps_geomean_pe, ps_dict['geomean'])
# average(ps_mean_year, ps_mean_pe, ps_dict['mean'])
# average(ps_median_year, ps_median_pe, ps_dict['median'])

# # Update plotting calls
# plot_grouped_metrics(nf_dict, 'mean', 'nf')
# plot_grouped_metrics(nf_dict, 'median', 'nf')
# plot_grouped_metrics(ps_dict, 'mean', 'ps', format_order=1)
# plot_grouped_metrics(ps_dict, 'median', 'ps', format_order=1)
# plot_grouped_metrics(nf_dict, 'mean', 'nf_year')
# plot_grouped_metrics(nf_dict, 'median', 'nf_year')
# plot_grouped_metrics(ps_dict, 'mean', 'ps_year', format_order=1)
# plot_grouped_metrics(ps_dict, 'median', 'ps_year', format_order=1)
