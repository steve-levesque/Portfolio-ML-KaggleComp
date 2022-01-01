import numpy as np
import pandas as pd
import os


def save_data(submission_file, y_pred, path, file_name, ext=".csv"):
    submission_file.iloc[:, 1] = y_pred
    submission_file.to_csv(path + file_name + ext, index=False)
