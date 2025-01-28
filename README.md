This API is used to return a approximated runtime given specified model parameters within the API. To interact with the API follow the code examples in the examples directory.
To create your own query, you can follow the examples in example queries.

The configuration for a query.yaml file to pass to the API follows as
The data is initially pulled from LLM survey.xlsx, but can also be loaded from checkpoints

query:
    year: int (can handle any value, as it has bounds given input data from the excel file)
    numerical_format: str (options: '32-16', '16-8', '16-4') Numerical format does not need to be specified, as it is also grouped by year in the 'configurations/example_configuration.yaml'
    phase: str (options: 'training', 'inference') Phase does not need to be specified. If it is not, it will default to inference as some internal calculations need this, but will not group the parallel strategy speedup by phase, so computed data can be different by explicitly denoting inference
    parallel_strategy: str (options: 'pipeline', 'tensor', 'automated search') Parallel Strategy does not need to be denoted. If it is not, it will default to the year grouping denoted in 'configurations/example_configuration.yaml'.
    chip_type: str (options: 'gpu', 'acc', 'wse') Chip type does not need to be specified, if it is not it will defualt to gpu.
    optimizer: str (options: 'sgd', 'rmsprop', 'adam') Optimizer does not need to be specified, if it is not it will default is adam

The configuration file, which can be found at 'configurations/example_configuration.yaml', is advised to not be changed. The year ranges can be changed, but nothing can be removed. The optimizers can be changed, but this would result in having to adjust internal code to change defaults and other interactions. Given this, it is advised not to change the defualt configuration. 
If the excel file path is changed, then changing this to max the correct filepath will not have any adverse effect to the api, and is recommended.
If you never change the configuration file path, you should be able to continue to use the same checkpoint file. A checkpoint file is not needed, as the csv files are small enough to not add any latency in initialization, but making the checkpoint allows you to save the .log file, and reuse that .log file.

The model returns a dictionary containing the output model size (TB), numerical format speedup, parallel strategy speedup, throughput (TFLOPS), and runtime (GPUH). It records the inputs and outputs to a log file if this is enabled.

Setup:

To setup, run 
pip install . 
in the sustainable_computing_workload directory

Run:

To run the first example, run python -m examples.single_query in the sustainable_computing_workload directory

Testing:

To automate building a query, you can look at the build_test.py in the examples directory. This gives an example of how to build many queries at once and save them to a yaml file. Each key must be an accepted key by the api. The values for each key is a list of the parameters you want to pass.