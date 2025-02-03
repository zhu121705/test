import pandas as pd
from collections import OrderedDict
from query.utils import read_yaml
from query import query_api
import random

def prev_year_range(year, year_ranges):
    if year <= year_ranges[0][1]:
        return [None]
    elif year >= year_ranges[-1][0]:
        return year_ranges[-2]
    else:
        for i in range(len(year_ranges)):
            if year <= year_ranges[i][1]:
                return year_ranges[i - 1]
            
def prev_parameter(param, param_options):
    for i in range(len(param_options)):
        if param_options[i] == param:
            if i == 0:
                return None
            return param_options[i - 1]

def validation(function, numerical_format_row, parallel_strategy_row):
    nf_year = numerical_format_row['Year']
    ps_year = parallel_strategy_row['Year']
    numerical_format = numerical_format_row['Numerical Format']
    parallel_strategy = parallel_strategy_row['Parallel strategy'].lower()
    parallel_strategy = 'automated search' if not (parallel_strategy == 'pipeline' or parallel_strategy == 'tensor') else parallel_strategy
    nf_speedup = numerical_format_row['Speedup (X)']
    ps_speedup = parallel_strategy_row['Speedup (X)']

    config_dict = read_yaml('configurations/example_configuration.yaml')['configuration']
    year_range_lists = config_dict['year_ranges']
    nf_year_ranges = year_range_lists['numerical_format']
    ps_year_ranges = year_range_lists['parallel_strategy']

    nf_prev_year = random.choice(prev_year_range(nf_year, nf_year_ranges))
    ps_prev_year = random.choice(prev_year_range(ps_year, ps_year_ranges))

    nf_options = ['32-16', '16-8', '16-4']
    pf_options = ['pipeline', 'tensor', 'automated search']

    numerical_format_prev = prev_parameter(numerical_format, nf_options)
    parallel_strategy_prev = prev_parameter(parallel_strategy, pf_options)

    # Create base dictionaries
    nf_queries = OrderedDict({'queries': OrderedDict()})
    ps_queries = OrderedDict({'queries': OrderedDict()})

    # Add current year queries
    nf_queries['queries']['query_1'] = OrderedDict({'year': nf_year})
    nf_queries['queries']['query_2'] = OrderedDict({
        'year': nf_year,
        'numerical_format': numerical_format
    }) if numerical_format is not None else None

    # Add previous year queries only if both year and parameter exist
    if nf_prev_year is not None and nf_prev_year != 'None':
        nf_queries['queries']['prev_query_1'] = OrderedDict({'year': nf_prev_year})
        if numerical_format_prev is not None:
            nf_queries['queries']['prev_query_2'] = OrderedDict({
                'year': nf_prev_year,
                'numerical_format': numerical_format_prev
            })

    # Add current year queries for parallel strategy
    ps_queries['queries']['query_1'] = OrderedDict({'year': ps_year})
    ps_queries['queries']['query_2'] = OrderedDict({
        'year': ps_year,
        'parallel_strategy': parallel_strategy
    }) if parallel_strategy is not None else None

    # Add previous year queries only if both year and parameter exist
    if ps_prev_year is not None and ps_prev_year != 'None':
        ps_queries['queries']['prev_query_1'] = OrderedDict({'year': ps_prev_year})
        if parallel_strategy_prev is not None:
            ps_queries['queries']['prev_query_2'] = OrderedDict({
                'year': ps_prev_year,
                'parallel_strategy': parallel_strategy_prev
            })

    # Clean up None values from queries
    nf_queries['queries'] = OrderedDict({k: v for k, v in nf_queries['queries'].items() if v is not None})
    ps_queries['queries'] = OrderedDict({k: v for k, v in ps_queries['queries'].items() if v is not None})

    query_object = query_api.Query('configurations/example_configuration.yaml', function=function)

    nf_queries = query_object.queries(queries_dict = nf_queries)
    ps_queries = query_object.queries(queries_dict = ps_queries)

    nf_year_speedup = nf_speedup * (nf_queries['queries']['prev_query_1']['numerical format speedup'] if 'prev_query_1' in nf_queries['queries'] else 1)
    nf_speedup *= nf_queries['queries']['prev_query_2']['numerical format speedup'] if 'prev_query_2' in nf_queries['queries'] else 1
    nf_query_year_speedup = nf_queries['queries']['query_1']['numerical format speedup']
    nf_query_speedup = nf_queries['queries']['query_2']['numerical format speedup']

    ps_year_speedup = ps_speedup * (ps_queries['queries']['prev_query_1']['parallel strategy speedup'] if 'prev_query_1' in ps_queries['queries'] else 1)
    ps_speedup *= ps_queries['queries']['prev_query_2']['parallel strategy speedup'] if 'prev_query_2' in ps_queries['queries'] else 1
    ps_query_year_speedup = ps_queries['queries']['query_1']['parallel strategy speedup']
    ps_query_speedup = ps_queries['queries']['query_2']['parallel strategy speedup']

    nf_year_prev_query = nf_queries['queries']['prev_query_1']['numerical format speedup'] if 'prev_query_1' in nf_queries['queries'] else -1
    nf_prev_query = nf_queries['queries']['prev_query_2']['numerical format speedup'] if 'prev_query_2' in nf_queries['queries'] else -1
    ps_year_prev_query = ps_queries['queries']['prev_query_1']['parallel strategy speedup'] if 'prev_query_1' in ps_queries['queries'] else -1
    ps_prev_query = ps_queries['queries']['prev_query_2']['parallel strategy speedup'] if 'prev_query_2' in ps_queries['queries'] else -1

    nf_year_pe = abs((nf_query_year_speedup - nf_year_speedup)/nf_year_speedup) * 100
    nf_pe = abs((nf_query_speedup - nf_speedup)/nf_speedup) * 100
    ps_year_pe = abs((ps_query_year_speedup - ps_year_speedup)/ps_year_speedup) * 100
    ps_pe = abs((ps_query_speedup - ps_speedup)/ps_speedup) * 100

    nf_year_ae = abs(nf_query_year_speedup - nf_year_speedup)
    nf_ae = abs(nf_query_speedup - nf_speedup)
    ps_year_ae = abs(ps_query_year_speedup - ps_year_speedup)
    ps_ae = abs(ps_query_speedup - ps_speedup)

    nf_dict = OrderedDict({
        'nf_format': numerical_format,
        'nf_year': nf_year,
        'nf_speedup': nf_speedup,
        'nf_query_speedup': nf_query_speedup,
        'year_speedup': nf_year_speedup,
        'year_query_speedup': nf_query_year_speedup,
        'nf_pe': nf_pe,
        'year_pe': nf_year_pe,
        'nf_ae': nf_ae,
        'year_ae': nf_year_ae,
        'nf_prev_query': nf_prev_query,
        'nf_year_prev_query': nf_year_prev_query
    })
    ps_dict = OrderedDict({
        'ps_strategy': parallel_strategy,
        'ps_year': ps_year,
        'ps_speedup': ps_speedup,
        'ps_query_speedup': ps_query_speedup,
        'year_speedup': ps_year_speedup,
        'year_query_speedup': ps_query_year_speedup,
        'ps_pe': ps_pe,
        'year_pe': ps_year_pe,
        'ps_ae': ps_ae,
        'year_ae': ps_year_ae,
        'ps_prev_query': ps_prev_query,
        'ps_year_prev_query': ps_year_prev_query
    })

    return nf_dict, ps_dict

# Main execution
numerical_format_df = pd.read_csv('csv/numerical_format.csv')
parallel_strategy_df = pd.read_csv('csv/parallel_strategy.csv')

nf_ps_csv = pd.DataFrame(columns=['function', 'percent_error', 'standard_deviation', 'average_error'])

for function in ['mean', 'median', 'geomean']:
    nf_csv = pd.DataFrame(columns=['nf_speedup', 'nf_query_speedup', 'year_speedup', 'year_query_speedup', 'nf_pe', 'year_pe'])
    ps_csv = pd.DataFrame(columns=['ps_speedup', 'ps_query_speedup', 'year_speedup', 'year_query_speedup', 'ps_pe', 'year_pe'])

    # Process numerical format rows
    for _, nf_row in numerical_format_df.iterrows():
        nf_dict, _ = validation(function, nf_row, parallel_strategy_df.iloc[0])  # Use any PS row as it won't affect NF results
        nf_csv = pd.concat([nf_csv, pd.DataFrame([nf_dict])], ignore_index=True)
    
    # Process parallel strategy rows
    for _, ps_row in parallel_strategy_df.iterrows():
        _, ps_dict = validation(function, numerical_format_df.iloc[0], ps_row)  # Use any NF row as it won't affect PS results
        ps_csv = pd.concat([ps_csv, pd.DataFrame([ps_dict])], ignore_index=True)

    nf_average_pe = nf_csv['nf_pe'].mean()
    nf_average_std = nf_csv['nf_pe'].std()
    nf_average_ae = nf_csv['nf_ae'].mean()
    nf_year_average_pe = nf_csv['year_pe'].mean()
    nf_year_average_std = nf_csv['year_pe'].std()
    nf_year_average_ae = nf_csv['year_ae'].mean()
    ps_average_pe = ps_csv['ps_pe'].mean()
    ps_average_std = ps_csv['ps_pe'].std()
    ps_average_ae = ps_csv['ps_ae'].mean()
    ps_year_average_pe = ps_csv['year_pe'].mean()
    ps_year_average_std = ps_csv['year_pe'].std()
    ps_year_average_ae = ps_csv['year_ae'].mean()

    nf_csv.to_csv('testing/nf_ps_csv/nf_results_' + function + '.csv', index=False, float_format='%.6f')
    ps_csv.to_csv('testing/nf_ps_csv/ps_results_' + function + '.csv', index=False, float_format='%.6f')

    nf_ps_csv = pd.concat([nf_ps_csv, pd.DataFrame([['nf_' + function, nf_average_pe, nf_average_std, nf_average_ae]], columns=nf_ps_csv.columns)], ignore_index=True)
    nf_ps_csv = pd.concat([nf_ps_csv, pd.DataFrame([['ps_' + function, ps_average_pe, ps_average_std, ps_average_ae]], columns=nf_ps_csv.columns)], ignore_index=True)

nf_ps_csv.to_csv('testing/nf_ps_results.csv', index=False, float_format='%.6f')