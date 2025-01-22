import math
from query.query_interface import SubQueryList

class HardwareComparison:
    """
    This class constructs a hardware comparison object as well as defines getters
    Inputs:
    - year: year the hardware was released
    - chip: name of the chip
    - vendor: name of the vendor
    - chip_type: type of chip (gpu, acc, wse)
    - transistor_count: number of transistors on the chip
    - area: area of the chip in mm^2
    - core_count: number of cores on the chip
    - percent_yield: percent yield of the chip
    - tdp: thermal design power of the chip
    - fp4: teraflops for 4-bit floating point
    - fp8: teraflops for 8-bit floating point
    - fp16_bf16: teraflops for 16-bit floating point (includes bfloat as well)
    - fp32: teraflops for 32-bit floating point
    - on_chip_mem: on-chip memory in GB
    - on_chip_bandwidth: on-chip bandwidth in GB/s
    - off_chip_mem: off-chip memory in GB
    - off_chip_bandwidth: off-chip bandwidth in GB/s
    - link: link to the source of the data
    """
    def __init__(self, year: int, chip: str, vendor: str, chip_type: str, transistor_count: int, area: int, core_count: int, percent_yield: int, tdp: int, fp4: float, fp8: float, fp16_bf16: float, fp32: float, on_chip_mem: int, on_chip_bandwidth: int, off_chip_mem: int, off_chip_bandwidth: int, link: str):
        self.year = year
        self.chip = chip
        self.vendor = vendor
        self.chip_type = chip_type
        self.transistor_count = transistor_count
        self.area = area
        self.core_count = core_count
        self.percent_yield = percent_yield
        self.tdp = tdp
        self.fp4_tflops = fp4
        self.fp8_tflops = fp8
        self.fp16_bf16_tflops = fp16_bf16
        self.fp32_tflops = fp32
        self.on_chip_mem = on_chip_mem
        self.on_chip_bandwidth = on_chip_bandwidth
        self.off_chip_mem = off_chip_mem
        self.off_chip_bandwidth = off_chip_bandwidth
        self.link = link

    #region Getters
    def get_year(self):
        return self.year
    
    def get_chip(self):
        return self.chip

    def get_chip_type(self):
        return self.chip_type
    
    def get_tflops(self, numerical_format):
        if numerical_format == 4:
            return self.fp4_tflops if not math.isnan(self.fp4_tflops) else None
        elif numerical_format == 8:
            return self.fp8_tflops if not math.isnan(self.fp8_tflops) else None
        elif numerical_format == 16:
            return self.fp16_bf16_tflops if not math.isnan(self.fp16_bf16_tflops) else None
        elif numerical_format == 32:
            return self.fp32_tflops if not math.isnan(self.fp32_tflops) else None
    #endregion

class HardwareComparisonList(SubQueryList):
    """
    This class constructs a list of hardware comparison objects as well as defines a query function
    Inputs:
    - input_dict: dictionary containing hardware comparison information
    """
    def __init__(self, input_dict: dict):
        self.hardware_comparison_list = []
        for sub_dict in input_dict.values():
            self.hardware_comparison_list.append(HardwareComparison(**sub_dict))
        # minimum gpu years for each chip type option. If a query year is less than the minimum year, the query year will be set to the minimum year
        self.min_year = {
            'gpu': min([item.get_year() for item in self.hardware_comparison_list if item.get_chip_type() == 'gpu']),
            'acc': min([item.get_year() for item in self.hardware_comparison_list if item.get_chip_type() == 'acc']),
            'wse': min([item.get_year() for item in self.hardware_comparison_list if item.get_chip_type() == 'wse'])
        }
        
    # Queries the hardware comparison list for the average tflops
    # Inputs:
    # - query_year: year to query
    # - query_chip_type: chip type to query
    # - query_numerical_format: numerical format to query
    # Outputs:
    # - average tflops
    def query(self, query_year: int, query_chip_type = None, query_numerical_format = None):
        tflops_list = []
        hardware_year = 0

        if query_chip_type in self.min_year and query_year < self.min_year[query_chip_type]:
            hardware_year = self.min_year[query_chip_type]
        else:
            for item in self.hardware_comparison_list:
                if query_chip_type == item.get_chip_type() and item.get_year() <= query_year and hardware_year < item.get_year():
                        hardware_year = item.get_year()

        for item in self.hardware_comparison_list:
            if query_chip_type == item.get_chip_type() and item.get_year() == hardware_year:
                tflops = item.get_tflops(query_numerical_format)
                if not isinstance(tflops, type(None)):
                    tflops_list.append(tflops)

        return self.average(tflops_list) if tflops_list else None 