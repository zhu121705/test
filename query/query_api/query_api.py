from query.hardware_configuration import hardware_comparison
from query.model_size import model_size
from query.numerical_format import numerical_format
from query.parallel_strategy import parallel_strategy
from query.utils.utils import clean_dictionary_keys, create_log, create_checkpoint
from query.utils.utils import read_csv_dir, read_yaml, write_yaml, write_txt, write_pkl, read_pkl
from collections import OrderedDict

class Query: 
    """
    Initializes a Query object from a yaml file or a checkpoint file. Query object creates query database and controls interation with the database.
    Args:
        path: Path to yaml file or checkpoint file
        verbose: Enables logging
        function: Used for interal averaging.
    """
    def __init__(self, path: str, verbose: bool = False, function = None):
        self.path = path
        self.function = function
        if self.path.endswith('.yaml'):
            if verbose: print('Loading query object from csv file path ' + path)
            self.load_excel(**read_yaml(path)['configuration'])
        else:
            if verbose: print('Loading query object from checkpoint path ' + path)
            self.load_pkl(path)

    def load_excel(self, path: str, year_ranges: OrderedDict, optimizers: OrderedDict = None, baseline_model: OrderedDict = None):
        """
        Loads from a yaml file with a csv directory path
        Args:
            path: Path to yaml file
            year_ranges: Dictionary of year ranges
            optimizers: Dictionary of optimizers
            baseline_model: Dictionary of baseline model parameters
        """
        self.input_dict = clean_dictionary_keys(read_csv_dir(path))
        self.year_ranges_dict = year_ranges
        self.optimizer_dict = optimizers
        self.baseline_model = baseline_model
        self.model_size_list = model_size.ModelSizeList(self.input_dict['model_size'], self.year_ranges_dict['model_size'], optimizer_dict=self.optimizer_dict)
        if self.function:
            self.numerical_format_list = numerical_format.NumericalFormatList(self.input_dict['numerical_format'], self.year_ranges_dict['numerical_format'], self.function)
            self.parallel_strategy_list = parallel_strategy.ParallelStrategyList(self.input_dict['parallel_strategy'], self.year_ranges_dict['parallel_strategy'], self.function)
        else:
            self.numerical_format_list = numerical_format.NumericalFormatList(self.input_dict['numerical_format'], self.year_ranges_dict['numerical_format'])
            self.parallel_strategy_list = parallel_strategy.ParallelStrategyList(self.input_dict['parallel_strategy'], self.year_ranges_dict['parallel_strategy'])
        self.hardware_comparison_list = hardware_comparison.HardwareComparisonList(self.input_dict['hardware_comparison'])
        self.log_file = create_log()

    def save_pkl(self, path: str = None):
        """
        Saves checkpoint to a pkl file
        Args:
            path: path to save checkpoint
        """
        if path is None and not self.path.endswith('.pkl'):
            path = create_checkpoint()
        else:
            path = self.path
        write_pkl(content=self, file=path)

    def load_pkl(self, path: str):
        """
        loads checkpoint from a pkl file
        Args:
            path: path to load checkpoint
        """
        loaded_obj = read_pkl(path)
        self.__dict__.update(loaded_obj.__dict__)
        self.path = path

    def query(self, query_path: str = None, query_dict: OrderedDict = None, log: bool = False, numerical_format_scaling: bool = True, parallel_strategy_scaling: bool = True, hardware_configuration_scaling: bool = True) -> OrderedDict:
        """
        queries each internal list for model paramaters and speedup calculations to approximate runtime
        Args:
            query_path: path to yaml file
            query_dict: dictionary of query parameters
            log: enables logging
            numerical_format_scaling: enables numerical format speedup
            parallel_strategy_scaling: enables parallel strategy speedup
            hardware_configuration_scaling: enables hardware configuration speedup
        Returns:
            output_dict: dictionary of query results
        """
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


        numerical_format_query = self.numerical_format_list.query(year, numerical_format, phase)
        parallel_strategy_query = self.parallel_strategy_list.query(year, parallel_strategy, phase)
        if self.function:
            model_size_query = self.model_size_list.query(year, phase, optimizer, self.function)
            hardware_comparison_query = self.hardware_comparison_list.query(year, chip_type, numerical_format_query['activation_numerical_format'], self.function)
        else:
            model_size_query = self.model_size_list.query(year, phase, optimizer)
            hardware_comparison_query = self.hardware_comparison_list.query(year, chip_type, numerical_format_query['activation_numerical_format'])

        if phase == 'inference': 
            baseline_runtime = (self.baseline_model['inference_runtime'] / (1000 * 60 * 60)) * (model_size_query['seq_length'] / self.baseline_model['sequence_length'])
            baseline_tflops = self.baseline_model['inference_tflops']  
        elif phase == 'training':
            baseline_runtime = self.baseline_model['training_runtime']
            baseline_tflops = self.baseline_model['training_tflops']

        try:
            runtime = baseline_runtime * (model_size_query['parameters'] / self.baseline_model['parameters'])
            runtime *= (baseline_tflops / hardware_comparison_query) if hardware_configuration_scaling else 1
            runtime *= (1 / numerical_format_query['speedup']) if numerical_format_scaling else 1
            runtime *= (1 / parallel_strategy_query) if parallel_strategy_scaling else 1
        except:
            runtime = None

        if not (isinstance(runtime, float) or isinstance(runtime, int)):
            runtime = None

        billion = 10**9 # Billion
        terabyte = 2**43 # Terabyte
        conversion = billion/terabyte 

        model_size_query['activation_size'] *= numerical_format_query['activation_numerical_format'] * conversion
        model_size_query['weight_size'] *= numerical_format_query['weight_numerical_format'] * conversion if phase == 'inference' else numerical_format_query['activation_numerical_format'] * conversion
        model_size_query['optimizer_size'] = model_size_query['optimizer_size'] * 32 * conversion # assume optimizer is always in fp32

        #runtime = (models_params/baseline_params) * (baseline_tflops/chip_tflops) * (1/numerical_format_speedup) * (1/parallel_strategy_speedup) * baseline_runtime

        output_dict = OrderedDict({
            'average model size (TB)': model_size_query['activation_size'] + model_size_query['weight_size'] + model_size_query['optimizer_size'],
            'numerical format speedup': numerical_format_query['speedup'],
            'parallel strategy speedup': parallel_strategy_query,
            'hardware (TFLOPS)': hardware_comparison_query,
            'runtime (GPUH)': runtime,
            'parameters': model_size_query['parameters'],
        })

        if log:
            write_txt(self.log_file, query_params_dict, output_dict)

        return output_dict
    
    def queries(self, query_path: str = None, queries_dict: OrderedDict = None, log: bool = False, function = 'mean') -> list[OrderedDict]:
        """
        Used to query multiple queries at a time
        Args:
            query_path: path to yaml file
            queries_dict: dictionary of queries
            log: enables logging
            function: function to use for averaging
        Returns:
            return_dict: dictionary of results of multiple queries as sub-dictionaries
        """
        if query_path:
            query_dict = read_yaml(query_path)
        elif queries_dict:
            query_dict = queries_dict

        if query_dict.get('query') is not None:
            raise ValueError('Invalid query.yaml configuration. Did you mean to call self.query() instead?')
        return_dict = OrderedDict({'queries': OrderedDict()})
        for key, value in query_dict['queries'].items():
            query_result = self.query(query_dict=value, log=log) if isinstance(value, dict) else self.query(query_path=value, log=log)
            return_dict['queries'][key] = query_result
        return return_dict
            