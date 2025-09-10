import os
import csv
import glob
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split


def csv_for_kfold(mode, num_folds, data, out_path):
  
  file_path = f"{out_path}/{mode}{num_folds}.csv"
  fields = ["filepath", "label"]

  write_header = not os.path.exists(file_path)

  with open(file_path, 'a', newline='') as file:
      writer = csv.DictWriter(file, fieldnames=fields)

      if write_header:
        writer.writeheader()

      writer.writerow(data)
      
def datasets_gen(out_path):
  # store relative image file path
  X = []
  # store labels with-stone=1 and without-stone=0
  y = []

  for label_path in os.listdir(out_path):
    # get labels from images folder
    label = 1 if label_path == 'with-stone' else 0

    # iterate over each file in the dataset
    for image_path in glob.glob(f"{out_path}/{label_path}/*png"):
      rel_path = os.path.relpath(image_path, out_path)

      X.append(rel_path)
      y.append(label)

  return X, y


def datasets_gen_kfold(X, y, num_folds, out_path):
  
  for num_folds in range(num_folds):

    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, stratify=y)

    data = {}

    # csv file for train data
    for feature, label in tqdm(zip(X_train, y_train), desc=f'Processing data for {num_folds+1} folds'):

      data['filepath'] = feature
      data['label'] = label

      csv_for_kfold(mode='train', num_folds=num_folds, data=data, out_path=out_path)

    # csv file for validation data
    for feature, label in zip(X_val, y_val):

      data['filepath'] = feature
      data['label'] = label

      csv_for_kfold(mode='val', num_folds=num_folds, data=data, out_path=out_path)