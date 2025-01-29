from query.query_interface import SubQueryList
from collections import OrderedDict
import math
import statistics

class ModelSize:
    """
    Defines a model size object, defines getters, and computes the maximum memory size of the model
    Args:
        year: year of the model
        model: name of the model
        company: company that created the model
        model_arch: architecture of the model (encoder, decoder, both)
        params: number of parameters in the model (in billions)
        seq_length: sequence length of the model
        layers: number of layers in the model
        d_model: model dimension
        d_ff: feed forward dimension
        n_head: number of heads
        n_kv_head: number of key-value heads
        max_kv_cache: maximum key-value cache
        training_flops: training flops of the model
        paper: paper the model is from
    """
    def __init__(self, year: int, model: str, company: str, model_arch: str, params: float, seq_length: int, layers: int, d_model: int, d_ff: int, n_head: int, n_kv_head: int, max_kv_cache: int, training_flops: float, paper: str, optimizers: dict):
        self.year = year
        self.model = model
        self.company = company
        self.model_arch = model_arch
        self.params = params
        self.seq_length = seq_length
        self.layers = layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.max_kv_cache = max_kv_cache
        self.training_flops = training_flops
        self.paper = paper
        self.inference_max_memory = self.get_model_max_size('inference')
        self.training_max_memory = OrderedDict({})
        for optimizer, value in optimizers.items():
            self.training_max_memory[optimizer] = self.get_model_max_size('training', value)

    # region Getters
    def get_model(self):
        return self.model

    def get_year(self):
        return self.year
    
    def get_params(self):
        return self.params
    
    def get_seq_length(self):
        return self.seq_length
    
    def get_layers(self):
        return self.layers
    
    def get_d_model(self):
        return self.d_model
    
    def get_d_ff(self):
        return self.d_ff
    
    def get_n_head(self):
        return self.n_head
    
    def get_max_kv_cache(self):
        return self.max_kv_cache
    
    def get_inference_max_memory(self):
        return self.inference_max_memory
    
    def get_training_max_memory(self, optimizer: str):
        return self.training_max_memory[optimizer]
    # endregion
    
    # Calculates the maximum memory size of the model loaded into memory given phase and optimizer
    # Inputs:
    # - phase: phase of the model (inference or training)
    # - optimizer: optimizer to query (Adam, RSMprop, SGD)
    # Outputs:
    # - dictionary containing the maximum memory size of the model for both activations and parameters (parameters include gradient and optimizer in training, and kvcache in inference)
    def get_model_max_size(self, phase: str = 'inference', optimizer: int = 0):
        """
        Calculates the maximum memory needed to load the model configuration into memory
        Args:
            phase: phase of the model ['inference', 'training']
            optimizer: optimizer to query as a ratio of parameters ['adam', 'rmsprop', 'sgd'] ex: adam = 3 (3x parameters needed to load adam optimizer)
        Returns:
            dictionary containing the maximum memory size of the model for activation, parameters, and optimizer
        """
        d_model_per_head = self.d_model / self.n_head

        #region ACTIVATIONS
        seq_len = self.seq_length if phase == 'training' or self.max_kv_cache == 0 else 1
        # projection layer
        initial_activation = seq_len * self.d_model
        attention_norm = seq_len * self.d_model
        q_prj = seq_len * self.d_model
        k_prj = seq_len * self.d_model * self.n_kv_head / self.n_head
        v_prj = seq_len * self.d_model * self.n_kv_head / self.n_head
        prj_output = attention_norm + q_prj + k_prj + v_prj
        prj_layer = max(initial_activation + attention_norm, prj_output)

        # self attention mechanism
        q = self.n_head * seq_len * d_model_per_head
        qkt = self.n_head * seq_len * self.seq_length
        softmax = self.n_head * seq_len * self.seq_length
        attn_v = seq_len * self.d_model
        proj_a = seq_len * self.d_model
        attn_output = qkt + softmax + attn_v + proj_a
        attn_layer = max(q + qkt, qkt + softmax, softmax + attn_v, attn_v + proj_a)

        # feed forward network
        ffn_input = seq_len * self.d_model
        ffn_norm = seq_len * self.d_model
        gate_proj = seq_len * self.d_ff
        up_proj = seq_len * self.d_ff
        ffn_actv = seq_len * self.d_ff
        down_proj = seq_len * self.d_model
        ffn_output = ffn_norm + gate_proj + up_proj + ffn_actv + down_proj
        ffn_layer = max(ffn_input + + ffn_norm, ffn_norm + gate_proj + up_proj, gate_proj + ffn_actv, ffn_actv + down_proj)
        max_inference_activations = max(prj_layer, attn_layer, ffn_layer) / 10**9
        training_activations = math.sqrt((prj_output + attn_output + ffn_output) * self.layers) / 10**9
        #endregion
  
        parameters = self.params 
        if phase == 'inference':
            return_dict = OrderedDict({'activation_size': max_inference_activations, 'weight_size': self.max_kv_cache + parameters, 'optimizer_size': 0, 'parameters': parameters, 'seq_length': self.seq_length})
        elif phase == 'training':
            return_dict = OrderedDict({'activation_size': training_activations, 'weight_size': parameters * 2, 'optimizer_size': optimizer * parameters, 'parameters': parameters, 'seq_length': self.seq_length})
        return return_dict

class ModelSizeList(SubQueryList):
    """
    Constructs a list of model size objects as well as defines a query function
    Args: 
        input_dict: dictionary of model size information
        year_ranges: list of tuples containing the start and end years of the year ranges
        optimizer_dict: dictionary of optimizers and their respective memory multipliers
    """
    def __init__(self, input_dict: dict, year_ranges: list[tuple], optimizer_dict: dict):
        self.model_size_list = []
        self.year_ranges = year_ranges
        for sub_dict in input_dict.values():
            self.model_size_list.append(ModelSize(**sub_dict, optimizers=optimizer_dict))

    # Queries the list of ModelSize objects for an average model size (Billions)
    # Inputs:
    # - query_year: year to query
    # - query_phase: phase to query (inference or training)
    # - query_optimizer: optimizer to query
    # Outputs:
    # - list of dictionaries containing the average model size (Billions) 
    def query(self, query_year: int = None, query_phase: str = 'inference', query_optimizer: str = 'adam', function = 'median'):
        """
        Queries the model size list for the average model size
        Args:
            query_year: year to query
            query_phase: phase to query ['inference', 'training']
            query_optimizer: optimizer to query ['adam', 'rmsprop', 'sgd']
            function: function to apply to model size list ['mean', 'median', 'geomean']
        Returns:
            list of dictionaries containing the average model size (Billions)
        """
        if query_optimizer == None:
            query_optimizer = 'adam'

        model_size_list = []

        year_lower, year_upper = self.select_year_index(query_year, self.year_ranges)

        for model_size_object in self.model_size_list:
            if year_lower <= model_size_object.get_year() <= year_upper or query_year is None:
                if query_phase == 'inference':
                    model_size_list.append(model_size_object.get_inference_max_memory())
                else:
                    model_size_list.append(model_size_object.get_training_max_memory(query_optimizer))
        if not model_size_list:
            return None
        
        # model_size_dict = OrderedDict({
        #     'activation_size': self.average([ms['activation_size'] for ms in model_size_list], function),
        #     'weight_size': self.average([ms['weight_size'] for ms in model_size_list], function),
        #     'optimizer_size': self.average([ms['optimizer_size'] for ms in model_size_list], function),
        #     'parameters': self.average([ms['parameters'] for ms in model_size_list], function),
        #     'seq_length': self.average([ms['seq_length'] for ms in model_size_list], function),
        # })

        return self.average(model_size_list, function)

    def average(self, model_size_list: list[float], function: str):
        model_size_dict = OrderedDict({
            'activation_size': 0,
            'weight_size': 0,
            'optimizer_size': 0,
            'parameters': 0,
            'seq_length': 0,
        })

        if function == 'mean':
            for ms in model_size_list:
                model_size_dict['activation_size'] += ms['activation_size']
                model_size_dict['weight_size'] += ms['weight_size']
                model_size_dict['optimizer_size'] += ms['optimizer_size']
                model_size_dict['parameters'] += ms['parameters']
                model_size_dict['seq_length'] += ms['seq_length']

            model_size_dict['activation_size'] /= len(model_size_list)
            model_size_dict['weight_size'] /= len(model_size_list)
            model_size_dict['optimizer_size'] /= len(model_size_list)
            model_size_dict['parameters'] /= len(model_size_list)
            model_size_dict['seq_length'] /= len(model_size_list)

        elif function == 'median':
            activation_size_list = [ms['activation_size'] for ms in model_size_list]
            weight_size_list = [ms['weight_size'] for ms in model_size_list]
            optimizer_size_list = [ms['optimizer_size'] for ms in model_size_list]
            parameters_list = [ms['parameters'] for ms in model_size_list]
            seq_length_list = [ms['seq_length'] for ms in model_size_list]

            model_size_dict['activation_size'] = statistics.median(activation_size_list)
            model_size_dict['weight_size'] = statistics.median(weight_size_list)
            model_size_dict['optimizer_size'] = statistics.median(optimizer_size_list)
            model_size_dict['parameters'] = statistics.median(parameters_list)
            model_size_dict['seq_length'] = statistics.median(seq_length_list)

        elif function == 'geomean':
            activation_size_list = [ms['activation_size'] for ms in model_size_list]
            weight_size_list = [ms['weight_size'] for ms in model_size_list]
            optimizer_size_list = [ms['optimizer_size'] for ms in model_size_list]
            parameters_list = [ms['parameters'] for ms in model_size_list]
            seq_length_list = [ms['seq_length'] for ms in model_size_list]

            model_size_dict['activation_size'] = statistics.geometric_mean(activation_size_list)
            model_size_dict['weight_size'] = statistics.geometric_mean(weight_size_list)
            model_size_dict['optimizer_size'] = statistics.geometric_mean(optimizer_size_list)
            model_size_dict['parameters'] = statistics.geometric_mean(parameters_list)
            model_size_dict['seq_length'] = statistics.geometric_mean(seq_length_list)

        return model_size_dict
