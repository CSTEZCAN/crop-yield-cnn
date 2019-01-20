import os


#
# MODEL CONSTANTS, PATHS AND FOLDERS
#

# Root for execution folder
EXEC_ROOT = os.getcwd()

# DBs
DATABASES_NAME = 'datasets'
DATABASES_DIR = os.path.join(EXEC_ROOT, DATABASES_NAME)
os.makedirs(DATABASES_DIR, exist_ok=True)

# Default DB
DB_NAME = 'source_dataset.db'
DB_PATH = os.path.join(EXEC_ROOT, DATABASES_DIR, DB_NAME)

# Models
MODELS_NAME = 'models'
MODELS_DIR = os.path.join(EXEC_ROOT, MODELS_NAME)
os.makedirs(MODELS_DIR, exist_ok=True)

# Models
RESULTS_NAME = 'results'
RESULTS_DIR = os.path.join(EXEC_ROOT, RESULTS_NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)
