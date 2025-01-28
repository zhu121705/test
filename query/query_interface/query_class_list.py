from abc import ABC, abstractmethod
from collections import OrderedDict
import statistics

class SubQueryList(ABC):
    @abstractmethod
    def query(self, input_dict: OrderedDict):
        pass

    def get_year_range_tuple(self, query_year):
        if query_year < self.year_ranges[0][0]:
            return self.year_ranges[0]
        elif query_year > self.year_ranges[-1][1]:
            return self.year_ranges[-1]
        
        previous_range = self.year_ranges[0]
        for year_range in self.year_ranges:
            if year_range[0] <= query_year <= year_range[1]:
                return year_range
            elif query_year < year_range[0]:
                return previous_range
            previous_range = year_range
        return None

    def average(self, speedup_list: list[float], function) -> float:
        if function == 'mean':
            return_avg = round(sum(speedup_list) / len(speedup_list), 6) if speedup_list else None
        elif function == 'median':
            return_avg =  round(statistics.median(speedup_list), 6) if speedup_list else None
        elif function == 'geomean':
            return_avg =  round(statistics.geometric_mean(speedup_list), 6) if speedup_list else None
        else:
            return_avg = round(sum(speedup_list) / len(speedup_list), 6) if speedup_list else None
        return return_avg

    def select_year_index(self, query_year: int, year_ranges: list[list]) -> list[int]:
        if query_year < year_ranges[0][0]:
            year_lower = year_ranges[0][0]
            year_upper = year_ranges[0][1]
        elif query_year > year_ranges[-1][1]:
            year_lower = year_ranges[-1][0]
            year_upper = year_ranges[-1][1]
        else:
            for i, year_range in enumerate(year_ranges):
                if year_range[0] <= query_year <= year_range[1]:
                    year_range_index = i # index year range list for year rangess
                    break
                elif query_year < year_range[0]:
                    year_range_index = i - 1
                    break

            year_lower = year_ranges[year_range_index][0]
            year_upper = year_ranges[year_range_index][1]

        return [year_lower, year_upper]