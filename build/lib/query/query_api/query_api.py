from query.hardware_configuration import hardware_comparison
from query.model_size import model_size
from query.numerical_format import numerical_format
from query.parallel_strategy import parallel_strategy
from query.utils.utils import clean_dictionary_keys, create_log, create_checkpoint
from query.utils.utils import read_excel, read_yaml, write_yaml, write_txt, write_pkl, read_pkl
from collections import OrderedDict

class Query: 
    def __init__(self, path: str, verbose: bool = False):
        self.path = path
        if self.path.endswith('.yaml'):
            if verbose: print('Loading query object from excel file path ' + path)
            self.load_excel(**read_yaml(path)['configuration'])
        else:
            if verbose: print('Loading query object from checkpoint path ' + path)
            self.load_pkl(path)

    def load_excel(self, excel_path: str, year_ranges: OrderedDict, optimizers: OrderedDict = None):
        self.input_dict = clean_dictionary_keys(read_excel(excel_path))
        self.year_ranges_dict = year_ranges
        self.optimizer_dict = optimizers
        self.model_size_list = model_size.ModelSizeList(self.input_dict['model_size'], self.year_ranges_dict['model_size'], optimizer_dict=self.optimizer_dict)
        self.numerical_format_list = numerical_format.NumericalFormatList(self.input_dict['numerical_format'], self.year_ranges_dict['numerical_format'])
        self.parallel_strategy_list = parallel_strategy.ParallelStrategyList(self.input_dict['parallel_strategy'], self.year_ranges_dict['parallel_strategy'])
        self.hardware_comparison_list = hardware_comparison.HardwareComparisonList(self.input_dict['hardware_comparison'])
        self.log_file = create_log()

    def save_pkl(self, path: str = None):
        if path is None and not self.path.endswith('.pkl'):
            path = create_checkpoint()
        else:
            path = self.path
        write_pkl(content=self, file=path)

    def load_pkl(self, path: str):
        loaded_obj = read_pkl(path)
        self.__dict__.update(loaded_obj.__dict__)
        self.path = path

    def query(self, query_path: str = None, query_dict: OrderedDict = None, log: bool = False) -> OrderedDict:
        
        if query_path is None and query_dict is None:
            raise ValueError('Either query_path or query_dict is required')
        elif query_dict is None:
            query_dict = read_yaml(query_path)
            if query_dict.get('queries') is not None: raise ValueError('Invalid query.yaml configuration. Did you mean to call self.queries() instead?')
            query_params_dict = query_dict['query']   
        else:
            query_params_dict = query_dict
            if query_params_dict.get('queries') is not None: raise ValueError('Invalid query.yaml configuration. Did you mean to call self.queries() instead?')

        year = query_params_dict.get('year')
        if year is None:
            raise ValueError('year is a required input')

        phase = query_params_dict.get('phase')
        numerical_format = query_params_dict.get('numerical_format')
        parallel_strategy = query_params_dict.get('parallel_strategy')
        optimizer = query_params_dict.get('optimizer')
        chip_type = query_params_dict.get('chip_type')
            
        model_size_query = self.model_size_list.query(year, phase, optimizer)
        numerical_format_query = self.numerical_format_list.query(year, numerical_format)
        parallel_strategy_query = self.parallel_strategy_list.query(year, parallel_strategy, phase)

        hardware_comparison_query = self.hardware_comparison_list.query(year, chip_type, numerical_format_query['activation_numerical_format'])

        model_size_query['activation_size'] *= numerical_format_query['activation_numerical_format'] * (10**12 / 2**40)
        model_size_query['weight_size'] *= numerical_format_query['weight_numerical_format'] * (10**9 / 2**40) if phase == 'inference' else numerical_format_query['activation_numerical_format'] * (10**9 / 2**30)
        model_size_query['optimizer_size'] = model_size_query['optimizer_size'] * 32 * (10**9 / 2**40)

        runtime = 1302.4 # bert runtime as placeholder

        output_dict = OrderedDict({
            'average model size (TB)': model_size_query['activation_size'] + model_size_query['weight_size'],
            'numerical format speedup': numerical_format_query['speedup'],
            'parallel strategy speedup': parallel_strategy_query,
            'hardware (TFLOPS)': hardware_comparison_query,
            'runtime (GPUH)': runtime
        })

        if log:
            write_txt(self.log_file, query_params_dict, output_dict)

        return output_dict
    
    def queries(self, path: str, log: bool = False) -> list[OrderedDict]:
        query_dict = read_yaml(path)
        query_results = []
        if query_dict.get('query') is not None:
            raise ValueError('Invalid query.yaml configuration. Did you mean to call self.query() instead?')
        for value in query_dict['queries'].values():
            query_result = self.query(query_dict=value, log=log) if isinstance(value, dict) else self.query(query_path=value, log=log)
            query_results.append(query_result)
        return query_results
            