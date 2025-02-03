from query.query_interface import SubQueryList
from collections import OrderedDict

class ParallelStrategy:
    """
    Initializes a parallel strategy object and define getters
    Args:
        year: input year
        work: the work to attribute the parallel strategy to
        company: the company that implemented the parallel strategy
        parallel_strategy: the parallel strategy used
        phase: the phase of the parallel strategy (inference or training)
        speedup: the speedup of the parallel strategy
        paper: the paper reference
    """
    def __init__(self, year: int, work: str, company: str, parallel_strategy: str, phase: str, speedup: float, paper: str):
        self.year = year
        self.work = work
        self.company = company
        self.parallel_strategy = parallel_strategy
        self.phase = phase
        self.speedup = speedup
        self.paper = paper

    #region Getters
    def get_year(self):
        return self.year
    
    def get_work(self):
        return self.work
    
    def get_parallel_strategy(self):
        return self.parallel_strategy
    
    def get_phase(self):
        return self.phase

    def get_speedup(self):
        return self.speedup
    #endregion

class ParallelStrategyList(SubQueryList):
    """
    Initializes a parallel strategy list, as well as define a method to query the list
    The input data is reformated and removes keys certain keys.
    Args:
        input_dict: dictionary of parallel strategy information
        year_ranges: list of tuples containing the start and end years of the year ranges
        function: function to calculate the average speedup
    """
    def __init__(self, input_dict: dict, year_ranges: list[tuple], function: str = 'geomean'):

        self.parallel_strategy_dict = OrderedDict()
        self.parallel_strategy_year_dict = OrderedDict({
            'year_ranges': OrderedDict({})
        })
        self.parallel_strategy_list = []
        self.year_ranges = year_ranges
        self.parallel_strategy_set = set()

        self.preprocess(input_dict)    
        self.compute_speedups(function)

    def query(self, query_year: int = None, query_parallel_strategy: str = None, query_phase: str = None):
        """
        Queries the list of ParallelStrategy objects for an average speedup
            query_year: year to query
            query_parallel_strategy: parallel strategy to query (pipeline, tensor, data, automated search)
            query_phase: phase to query (inference or training)    
        """
        first_parallel_strategy = next(iter(self.parallel_strategy_dict.values()))
        last_parallel_strategy = next(reversed(self.parallel_strategy_dict.values()))
        speedup_string = f'{query_phase}_speedup' if query_phase else 'speedup'

        if query_parallel_strategy:
            return self.parallel_strategy_dict[query_parallel_strategy][speedup_string]['average_speedup']
        elif query_year:
            year_range = self.get_year_range_tuple(query_year)
            year_range = (year_range[0], year_range[1])
            return self.parallel_strategy_year_dict['year_ranges'][year_range]['average_speedup']

        else:
            if query_year < first_parallel_strategy['year_range'][0]:
                return first_parallel_strategy[speedup_string]['average_speedup']
            elif query_year > last_parallel_strategy['year_range'][1]:
                return last_parallel_strategy[speedup_string]['average_speedup']
            
            previous_speedup = None
            for parallel_strategy_dict in self.parallel_strategy_dict.values():
                year_range = parallel_strategy_dict['year_range']
                if year_range[0] <= query_year <= year_range[1]:
                    return parallel_strategy_dict[speedup_string]['average_speedup']
                elif query_year < year_range[0]:
                    return previous_speedup
                previous_speedup = parallel_strategy_dict[speedup_string]['average_speedup']
            return None
        
    def preprocess(self, input_dict: dict):
        """
        Preprocesses the input dictionary remove invalid keys
        Args:
            input_dict: dictionary of parallel strategy information
        """
        prev_parallel_strategy = None
        for sub_dict in input_dict.values():
            item_parallel_strategy = sub_dict['parallel_strategy']
            item_year = sub_dict['year']
            item_phase = sub_dict['phase']
            
            # removes unnecesary keys that dont have a parallel strategy value
            if not isinstance(item_parallel_strategy, str):
                continue

            # adjust keys to fit within the parallel strategy set. Comment out if you want to keep the original keys from your input data, or adjust the comparison to remove and adjust input keys.
            if '/' in item_parallel_strategy or 'data' in item_parallel_strategy:
                item_parallel_strategy = 'automated search'

            # removes keys that are out of order. comment out if you plan to adjust average computation to take this out of order keys into account
            # by default, the average computation does not work without this if block
            if item_parallel_strategy in self.parallel_strategy_set:
                if item_parallel_strategy != prev_parallel_strategy:
                    continue
            else:
                self.parallel_strategy_set.add(item_parallel_strategy)
                prev_parallel_strategy = item_parallel_strategy

            parallel_strategy_obj = ParallelStrategy(**sub_dict)

            # add indexing to parallel strategy dictionary
            if item_parallel_strategy not in self.parallel_strategy_dict:
                self.parallel_strategy_dict[item_parallel_strategy] = OrderedDict({
                    'year_range': [item_year, item_year],
                    'training_speedup': OrderedDict({
                        'parallel_structure_objects': [],
                        'average_speedup': None
                    }),
                    'inference_speedup': OrderedDict({
                        'parallel_structure_objects': [],
                        'average_speedup': None
                    }),
                    'speedup': OrderedDict({
                        'parallel_structure_objects': [],
                        'average_speedup': None
                    })
                })
            self.parallel_strategy_dict[item_parallel_strategy]['year_range'] = [
                min(item_year, self.parallel_strategy_dict[item_parallel_strategy]['year_range'][0]),
                max(item_year, self.parallel_strategy_dict[item_parallel_strategy]['year_range'][1])
            ]
            self.parallel_strategy_dict[item_parallel_strategy]['speedup']['parallel_structure_objects'].append(parallel_strategy_obj)
            self.parallel_strategy_dict[item_parallel_strategy][f'{item_phase}_speedup']['parallel_structure_objects'].append(parallel_strategy_obj)
            
            year_range = self.get_year_range_tuple(item_year)
            year_range = (year_range[0], year_range[1])
            if year_range not in self.parallel_strategy_year_dict['year_ranges']:
                self.parallel_strategy_year_dict['year_ranges'][year_range] = [parallel_strategy_obj]
            else:
                self.parallel_strategy_year_dict['year_ranges'][year_range].append(parallel_strategy_obj)

    def compute_speedups(self, function: str = 'mean'):
        """
        Computes the average speedup for each parallel strategy and phase
        Args:
            function: function to calculate the average speedup
        """
        # Process parallel_strategy_dict
        for parallel_strategy_dict in self.parallel_strategy_dict.values():
            for key, sub_dict in parallel_strategy_dict.items():
                if 'speedup' in key:
                    speedup_list = [obj.get_speedup() for obj in sub_dict['parallel_structure_objects']]
                    parallel_strategy_dict[key]['average_speedup'] = self.average(speedup_list, function)

        self.parallel_strategy_dict = OrderedDict(
            sorted(self.parallel_strategy_dict.items(), key=lambda item: item[1]['year_range'])
        )

        # Process parallel_strategy_year_dict
        compounding_speedup = 1
        
        for year_range, objects in self.parallel_strategy_year_dict['year_ranges'].items():
            speedup_list = [obj.get_speedup() for obj in objects]
            avg_speedup = self.average(speedup_list, function) * compounding_speedup
            compounding_speedup = avg_speedup
            self.parallel_strategy_year_dict['year_ranges'][year_range] = {
                'parallel_structure_objects': objects,
                'average_speedup': avg_speedup
            }
        # Apply compounding speedups to parallel_strategy_dict
        compounding_speedup = 1
        for parallel_strategy_dict in self.parallel_strategy_dict.values():
            parallel_strategy_dict['speedup']['average_speedup'] *= compounding_speedup
            if parallel_strategy_dict['training_speedup']['average_speedup']:
                parallel_strategy_dict['training_speedup']['average_speedup'] *= compounding_speedup
            if parallel_strategy_dict['inference_speedup']['average_speedup']:
                parallel_strategy_dict['inference_speedup']['average_speedup'] *= compounding_speedup
            compounding_speedup = parallel_strategy_dict['speedup']['average_speedup']