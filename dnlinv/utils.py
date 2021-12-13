import sys
import os
import os.path


def setup_bart():
    bart_path = os.environ['TOOLBOX_PATH']
    if bart_path is None:
        raise EnvironmentError("BART toolbox not found. Please specify TOOLBOX_PATH environment variable.")
    if os.path.exists(bart_path) is not True:
        raise EnvironmentError(f"BART cannot be found at {bart_path}. Please double-check the TOOLBOX_PATH "
                               f"environment variable.")
    print(f'Using BART at {bart_path}')
    sys.path.append(os.path.join(bart_path, 'python'))  # Use bart toolbox path
