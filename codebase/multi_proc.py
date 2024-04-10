### Usage ###
# This function runs a given function in parallel using either threading or multiprocessing.
# func: The function to be executed in parallel.
# args_list: A list of argument tuples to be passed to func.
# max_workers: The maximum number of workers to use. Default is 4.
# use_thread: A boolean value indicating whether to use threading (if True) or multiprocessing (if False). Default is False.

### Dependencies ###
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

### Function ###
def run_in_parallel(func, args_list, max_workers=4, use_thread=False):
    # Choose the executor class based on the value of use_thread.
    if use_thread:
        executor_class = ThreadPoolExecutor
    else:
        executor_class = ProcessPoolExecutor

    # Create an instance of the chosen executor class with the specified number of workers.
    with executor_class(max_workers=max_workers) as executor:
        # Use the executor to map the function to the arguments and execute them in parallel.
        # The results are collected into a list.
        results = list(executor.map(func, args_list))
    # Return the list of results.
    return results