from query.query_interface import SubQueryList
from collections import OrderedDict

class NumericalFormat:
    """
    Initiliazes a numerical format object, and define getters
    Args:
        year: input year
        format: floating point format
        numerical_format: bitwidths of floating point format
        phase: training or inumerical_formaterence
        speedup: speedup of the numerical format
        paper: paper reference
    """
    def __init__(self, year: int, format: str, numerical_format: str, phase: str, speedup: float, paper: str):
        self.year = year
        self.format = format
        self.numerical_format = numerical_format
        self.activation_format = int(numerical_format.split('-')[0])
        self.weight_format = int(numerical_format.split('-')[1])
        self.phase = phase
        self.speedup = speedup
        self.paper = paper

    #region Getters
    def get_year(self):
        return self.year
    
    def get_format(self):
        return self.format
    
    def get_numerical_format(self):
        return self.numerical_format

    def get_phase(self):
        return self.phase

    def get_speedup(self):
        return self.speedup
    #endregion

    # redefine __str__ and __repr__ methods for internal validation
    def __str__(self):
        return f'(Year: {self.year}, Format: {self.numerical_format})'
    
    def __repr__(self):
        return self.__str__()


class NumericalFormatList(SubQueryList):
    """
    Initializes a numerical format list and defines a method to query the list
    Args:
        input_dict: dictionary of numerical format inumerical_formatormation
        year_ranges: list of tuples containing the start and end years of the year ranges
        function: function to calculate the average speedup
    """
    def __init__(self, input_dict: dict, year_ranges: list[tuple], function: str = 'geomean'):
        self.numerical_format_list = []
        self.year_ranges = [tuple(year_range) for year_range in year_ranges]
        self.numerical_format_set = set()

        for sub_dict in input_dict.values():
            self.numerical_format_set.add(sub_dict['numerical_format'])

        self.numerical_format_dict = OrderedDict({
            'year_grouping': OrderedDict(),
            'format_grouping': OrderedDict(),
        })
        
        for sub_dict in input_dict.values():
            self.numerical_format_list.append(NumericalFormat(**sub_dict))

        for year_range in self.year_ranges:
            self.numerical_format_dict['year_grouping'][year_range] = OrderedDict({
                'speedup_list': [],
                'average_speedup': 0
            })

        for sub_dict in input_dict.values():
            if sub_dict['numerical_format'] not in self.numerical_format_dict['format_grouping']:
                self.numerical_format_dict['format_grouping'][sub_dict['numerical_format']] = OrderedDict({
                    'speedup_list': [NumericalFormat(**sub_dict)],
                    'average_speedup': 0
                })
            else:
                self.numerical_format_dict['format_grouping'][sub_dict['numerical_format']]['speedup_list'].append(
                    NumericalFormat(**sub_dict)
                )
            self._add_to_year_grouping(sub_dict)

        for sub_dict in self.numerical_format_dict.values():
            prev_speedup = 1
            for grouping in sub_dict.values():
                speedup_list = [obj.get_speedup() for obj in grouping['speedup_list']]
                grouping['average_speedup'] = self.average(speedup_list, function) * prev_speedup
                prev_speedup = grouping['average_speedup']

    def _add_to_year_grouping(self, sub_dict: dict):
        """
        Add item to appropriate year grouping
        Args:
            sub_dict: dictionary containing numerical format parameters
        """
        for year_range in self.year_ranges:
            if year_range[0] <= sub_dict['year'] <= year_range[1]:
                self.numerical_format_dict['year_grouping'][year_range]['speedup_list'].append(
                    NumericalFormat(**sub_dict)
                )

    def _get_numerical_formats(self, year_range_tuple):
        """
        Get the activation and weight numerical formats for a given year range
        Args:
            year_range_tuple: tuple containing the start and end years of the year range
        Returns:
            Tuple containing the activation and weight numerical formats
        """
        activation_format = 32  # initialize to highest possible value
        weight_format = 32
        for obj in self.numerical_format_dict['year_grouping'][year_range_tuple]['speedup_list']:
            if weight_format > obj.weight_format:
                weight_format = obj.weight_format
                activation_format = obj.activation_format
        return activation_format, weight_format

    def query(self, query_year: int = None, numerical_format: str = None, phase: str = 'inference', funtion: str = 'mean'):
        """
        Queries the numerical format lists for the average numerical format speedup
        Args:
            query_year: Year to query for
            numerical_format: Specific format to query (e.g. '8-16')
            phase: phase to determine numerical format
            function: function to calculate the average speedup
        Returns:
            OrderedDict with speedup and format information
        """

        phase = 'inference' if phase == None else phase

        if numerical_format:
            activation_format, weight_format = map(int, numerical_format.split('-'))
            if phase == 'training':
                for nf in self.numerical_format_set:
                    if numerical_format.split('-')[0] == nf.split('-')[1]:
                        speedup = self.numerical_format_dict['format_grouping'][nf]['average_speedup']
                        break
                    else:
                        speedup = 1
                return OrderedDict({
                    'speedup': speedup,
                    'activation_numerical_format': activation_format,
                    'weight_numerical_format': weight_format
                })
            elif phase == 'inference':
                return OrderedDict({
                    'speedup': self.numerical_format_dict['format_grouping'][numerical_format]['average_speedup'],
                    'activation_numerical_format': activation_format,
                    'weight_numerical_format': weight_format
                })
        elif query_year:
            if phase == 'training':
                for year_range in self.year_ranges:
                    if query_year <= year_range[1]:
                        speedup = 1
                        break
                    if query_year >= year_range[0]:
                        speedup = self.numerical_format_dict['year_grouping'][year_range]['average_speedup']
                        break

            year_range_tuple = self.get_year_range_tuple(query_year)
            if not year_range_tuple:
                return None
            
            if phase == 'inference':
                speedup = self.numerical_format_dict['year_grouping'][year_range_tuple]['average_speedup']
            activation_format, weight_format = self._get_numerical_formats(year_range_tuple)
            
            return OrderedDict({
                'speedup': speedup,
                'activation_numerical_format': activation_format,
                'weight_numerical_format': weight_format
            })

        return None