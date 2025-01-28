import os, yaml, sys, re, datetime, pickle
import pandas as pd
from itertools import product
from collections import OrderedDict
from yamlordereddictloader import SafeDumper
from yamlordereddictloader import SafeLoader

def read_yaml(file: str) -> OrderedDict:
    return yaml.load(open(file), Loader=SafeLoader)

def write_yaml(file: str, content: str) -> None:
    """
    if file exists at filepath, overwite the file, if not, create a new file
    :param filepath: string that specifies the destination file path
    :param content: yaml string that needs to be written to the destination file
    :return: None
    """
    if os.path.exists(file):
        os.remove(file)
    create_dir(os.path.dirname(file))
    out_file = open(file, 'a')
    out_file.write(yaml.dump( content, default_flow_style= False, Dumper=SafeDumper))

def create_dir(directory: str) -> None:
    """
    Checks the existence of a directory, if does not exist, create a new one
    :param directory: path to directory under concern
    :return: None
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        sys.exit()

def clean_dictionary_keys(dictionary: OrderedDict) -> OrderedDict:
    for sub_dict in dictionary.values():
        for index, metrics in sub_dict.items():
            sub_dict[index] = replace_key_spaces(metrics)
            sub_dict[index] = remove_unnamed_keys(metrics)
            sub_dict[index] = remove_parenthesis(metrics)
    
    dictionary = replace_key_spaces(dictionary)
    return dictionary

def replace_key_spaces(dictionary: OrderedDict) -> OrderedDict:
    keys_to_change = [key for key in dictionary if ' ' in key]
    for key in keys_to_change:
        new_key = key.replace(' ', '_').replace('-', '_').replace('#_', '').replace('/', '_')
        dictionary[new_key] = dictionary.pop(key)
    return dictionary

def remove_unnamed_keys(dictionary: OrderedDict) -> OrderedDict:
    keys_to_remove = [key for key in dictionary if 'unnamed' in key.lower()]
    for key in keys_to_remove:
        dictionary.pop(key)
    return dictionary

def remove_parenthesis(dictionary: OrderedDict) -> OrderedDict:
    keys_to_change = [key for key in dictionary if '(' in key and ')' in key]
    for key in keys_to_change:
        new_key = re.sub(r'\(.*?\)', '', key).strip('_').strip(' ')
        dictionary[new_key] = dictionary.pop(key)
    return dictionary

def read_sheet(path: str, sheet_name: str) -> OrderedDict:
    df = pd.read_excel(path, engine='openpyxl', sheet_name=sheet_name)

    sheet_dict = OrderedDict()
    for row_idx, row in df.iterrows():
        if row.isna().all():
            break

        row_dict = {}
        for col, value in row.items():
            row_dict[col.lower()] = value.lower() if isinstance(value, str) else value
        sheet_dict[row_idx] = row_dict
    
    return sheet_dict

def read_sheet_names(path: str) -> str:
    return pd.ExcelFile(path).sheet_names

def read_excel(path: str) -> OrderedDict:
    file_dict = OrderedDict()

    sheets = pd.ExcelFile(path).sheet_names

    for sheet in sheets:
        if sheet == 'reference':
            break
        file_dict[sheet] = read_sheet(path, sheet)

    return file_dict

def write_txt(path: str, params_dict, output_dict: OrderedDict):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, 'a') as file:
        file.write(f'{current_time}\t\t')
        for key, value in params_dict.items():
            file.write(f'{key}: {value}\t\t')
        file.write('\n\t\t\t\t\t\t')
        for key, value in output_dict.items():
            if isinstance(value, float):
                if key == 'runtime (GPUH)':
                    file.write(f'{key}: {value:.8f}\t\t')
                else:
                    file.write(f'{key}: {value:.6f}\t\t')
            else:
                file.write(f'{key}: {value}\t\t')
        file.write('\n\n')

def write_list_txt(path: str, input_list: list[OrderedDict]):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, 'a') as file:
        for dictionary in input_list:
            file.write(f'{current_time}\t\t')
            for key, value in dictionary.items():
                if isinstance(value, float):
                    file.write(f'{key}: {value:.6f}\t\t')
                else:
                    file.write(f'{key}: {value}\t\t')
            file.write('\n')

def create_log():
    dir = 'runs'
    if not os.path.exists(dir):
        os.makedirs(dir)
    files = os.listdir(dir)
    file_list = [file for file in files if os.path.isfile(os.path.join(dir, file)) and file.endswith('.log')]
    if file_list == []:
        return dir + '/run_0.log'

    pattern = re.compile(r'^run_\d+\.log$')
    matching_files = [file for file in file_list if pattern.match(file)]
    index = max([int(re.sub(r'run_', '', re.sub(r'.log', '', file))) for file in matching_files])
    return 'runs/run_' + str((index) + 1) + '.log'


def create_checkpoint():
    dir = 'checkpoints'
    if not os.path.exists(dir):
        os.makedirs(dir)
    files = os.listdir(dir)
    file_list = [file for file in files if os.path.isfile(os.path.join(dir, file))]
    if file_list == []:
        return dir + '/checkpoint_0.pkl'

    pattern = re.compile(r'^checkpoint_\d+\.pkl$')
    matching_files = [file for file in file_list if pattern.match(file)]
    index = max([int(re.sub(r'checkpoint_', '', re.sub(r'.pkl', '', file))) for file in matching_files])
    return dir + '/checkpoint_' + str((index) + 1) + '.pkl'

def write_pkl(content :object, file: str):
    if os.path.exists(file):
        os.remove(file)
    create_dir(os.path.dirname(file))
    out_file = open(file, 'wb')
    pickle.dump(content, out_file)

def read_pkl(file: str) -> object:
    in_file = open(file, 'rb')
    return pickle.load(in_file)

def build_test(test_dict: OrderedDict, path: str = 'example_queries/test.yaml'):
    key_options = ['year', 'phase', 'numerical_format', 'parallel_strategy', 'optimizer', 'chip_type']

    # Validate keys
    for key in test_dict:
        if key not in key_options:
            raise ValueError(f'Invalid key: {key} utils.build_test()')

    # Generate all combinations
    keys = test_dict.keys()
    values = test_dict.values()
    combinations = list(product(*values))

    # Create queries dictionary
    queries = OrderedDict({'queries': OrderedDict()})
    for i, combination in enumerate(combinations):
        query_dict = OrderedDict()
        for key, value in zip(keys, combination):
            query_dict[key] = value
        queries['queries'][f'query_{i+1}'] = query_dict

    # Write to yaml file
    write_yaml(path, queries)

def read_csv_dir(dir: str) -> OrderedDict:
    files = os.listdir(dir)
    file_list = [file for file in files if os.path.isfile(os.path.join(dir, file)) and file.endswith('.csv')]
    
    csv_dict = OrderedDict()

    for file in file_list:
        df = pd.read_csv(os.path.join(dir, file))
        sub_dict = OrderedDict({})

        for index, row in df.iterrows():
            if row.isna().all():
                break
            if index not in sub_dict:
                sub_dict[index] = OrderedDict()
            for header in df.columns:
                header_lower = header.lower()
                if 'Unnamed' in header:
                    continue
                if header_lower not in sub_dict[index]:
                    sub_dict[index][header_lower] = row[header].lower() if isinstance(row[header], str) else row[header]
        csv_dict[file.strip('.csv')] = sub_dict
    return csv_dict