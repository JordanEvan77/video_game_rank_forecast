from setup.setups import dir_base
import numpy as np
import time
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pickle
start_time = time.time()