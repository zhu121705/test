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
    years = [float(year) for year in year_dict.keys()]
    values = list(year_dict.values())

    if isinstance(values[0], list):
        extended_years = []
        flattened_values = []
        for year, value in zip(years, values):
            extended_years.extend([year] * len(value))
            flattened_values.extend(value)

        years = extended_years
        values = flattened_values

    plt.plot(years, values, label=label, marker='o', 
             markersize=8, linestyle='-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Parameters (B)', fontsize=12)
    plt.title(f'{label} Model Size Trend', fontsize=14)
    plt.xlim(2016, 2025)
    plt.legend(fontsize=10)
    plt.savefig(os.path.join(output_dir, f'{label}_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_multiple(data_list, title="Model Size Trends", filename="combined_plot"):
    fig, ax = plt.subplots(figsize=(20, 10))
    
    trend_colors = ['#3366cc', '#dc3912', '#109618']
    raw_data_color = '#ff9900'
    raw_data_alpha = 0.75
    
    legend_handles = []
    legend_labels = []
    
    for label, year_dict in data_list:
        if label == 'Raw Data':
            years = [float(year) for year in year_dict.keys()]
            values = list(year_dict.values())
            
            if isinstance(values[0], list):
                # Data grouping for boxplot
                range_data = {}
                for year_range in config['year_ranges']['model_size']:
                    start, end = year_range
                    mid_point = (start + end) / 2
                    range_data[mid_point] = []
                    for year, value_list in zip(years, values):
                        if start <= year <= end:
                            range_data[mid_point].extend(value_list)
                
                # Scatter plot
                extended_years = []
                flattened_values = []
                for year, value in zip(years, values):
                    extended_years.extend([year] * len(value))
                    flattened_values.extend(value)
                
                scatter = ax.scatter(extended_years, flattened_values, 
                                  s=50, c=raw_data_color, 
                                  alpha=raw_data_alpha, zorder=1)
                legend_handles.append(scatter)
                legend_labels.append(label)
                
                # Add boxplot
                positions = sorted(range_data.keys())
                box_data = [range_data[pos] for pos in positions]
                
                bp = ax.boxplot(box_data, positions=positions, 
                              widths=0.5,
                              whis=[0, 100],
                              patch_artist=True,
                              showfliers=False,
                              zorder=2)
                
                for element in ['boxes', 'whiskers', 'means', 'medians', 'caps']:
                    plt.setp(bp[element], color='#404040')
                plt.setp(bp['boxes'], facecolor='#404040', alpha=0.3)
                plt.setp(bp['medians'], color='#404040', linewidth=1)

    trend_idx = 0
    for label, year_dict in data_list:
        if label != 'Raw Data':
            years = [float(year) for year in year_dict.keys()]
            values = list(year_dict.values())
            
            line = ax.plot(years, values,
                         linestyle='-', linewidth=3,
                         c=trend_colors[trend_idx], zorder=3)[0]
            legend_handles.append(line)
            legend_labels.append(label)
            trend_idx += 1
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Parameters (B)', fontsize=14)
    plt.title(title, fontsize=16, pad=20)
    
    # Keep the year range indicators and dual axis
    all_years = sorted(list(plot_dict['raw_data'].keys()))
    plt.xlim(min(all_years) - 0.5, max(all_years) + 0.5)
    
    ax.set_xticks(all_years)
    ax.set_xticklabels([str(int(year)) for year in all_years], fontsize=10)
    
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    
    range_positions = []
    range_labels = []
    
    for year_range in config['year_ranges']['model_size']:
        start, end = year_range
        mid_point = (start + end) / 2
        range_positions.append(mid_point)
        range_labels.append(f'{start}-{end}' if start != end else str(start))
        
        ax.axvspan(start - 0.5, end + 0.5, 
                  color='white', alpha=0.1, zorder=0)
    
    ax2.set_xticks(range_positions)
    ax2.set_xticklabels(range_labels, fontsize=12)
    ax2.tick_params(length=10, pad=5)

    # Update legend with collected handles and labels
    ax.legend(legend_handles, legend_labels,
             fontsize=12, bbox_to_anchor=(1.05, 1), 
             loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename}.png'), 
                dpi=300, bbox_inches='tight')
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

    for year_key in plot_dict[year_query_function].keys():
        plot_dict[year_query_function][year_key] = queries['queries'][int(float(year_key))]['parameters']
    return plot_dict

plot_dict = OrderedDict({
    'raw_data': OrderedDict(),
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
        plot_dict['raw_data'][year] = [params]
        plot_dict['min_year'][year] = params
        plot_dict['max_year'][year] = params
        plot_dict['average_year'][year] = [params]
    else:
        plot_dict['raw_data'][year].append(params)
        plot_dict['min_year'][year] = min(plot_dict['min_year'][year], params)
        plot_dict['max_year'][year] = max(plot_dict['max_year'][year], params)
        plot_dict['average_year'][year].append(params)

for year in plot_dict['average_year']:
    plot_dict['average_year'][year] = sum(plot_dict['average_year'][year]) / len(plot_dict['average_year'][year])
plot_dict = query_function(plot_dict, 'mean')
plot_dict = query_function(plot_dict, 'median')
plot_dict = query_function(plot_dict, 'geomean')

plot_multiple([
    ('Raw Data', plot_dict['raw_data']),
    ('Arithmetic Mean', plot_dict['year_query_mean']),
    ('Median', plot_dict['year_query_median']),
    ('Geometric Mean', plot_dict['year_query_geomean'])
], title="Model Size Trend Comparison Over Time", filename="trends_comparison_boxplot")
