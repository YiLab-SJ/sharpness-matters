import os
import pandas as pd
import random
from pathlib import Path

HOME = Path(__file__).resolve().parent.parent

csv = pd.read_csv(
    "/research_jude/rgs01_jude/groups/pyigrp/projects/opendatashare/common/public_datasets/pneumothorax/vinbig/train_vin.csv"
)
# Select 500 non pneumothorax samples
image_ids = []
pos = []
print(csv.head())
print(f"Len of csv: {len(csv)}")
print(f"Unique imageids: {len(set(csv['image_id'].tolist()))}")
count = 0
for i in range(len(csv)):
    if csv.iloc[i]["class_name"] == "Pneumothorax":
        pos.append(csv.iloc[i]["image_id"])
        count += 1
    else:
        image_ids.append(csv.iloc[i]["image_id"])
pos = list(set(pos))
image_ids = list(set(image_ids))
random.seed(42)
randomlist = random.sample(image_ids, 594)  # 498 from ptx + 96 from vin
# If output doesn't exist, create it
os.makedirs(f"{HOME}/output", exist_ok=True)
# If output file exists, ask the user if they want to overwrite it
if os.path.exists(f"{HOME}/output/vin500neg.txt"):
    response = input(
        f"File {HOME}/output/vin500neg.txt already exists. Do you want to overwrite it? (y/n): "
    )
    if response.lower() != "y":
        print("User opted not to overwrite negative samples. Moving on")
else:
    with open(f"{HOME}/output/vin500neg.txt", "w+") as f:
        # write elements of list
        for items in randomlist:
            f.write("%s\n" % items)
        print("File written successfully")
    # close the file
    f.close()
# If output file exists, ask the user if they want to overwrite it
if os.path.exists(f"{HOME}/output/vin500pos.txt"):
    response = input(
        f"File {HOME}/output/vin500pos.txt already exists. Do you want to overwrite it? (y/n): "
    )
    if response.lower() != "y":
        print("User opted not to overwrite positive samples. Moving on")
else:
    with open(f"{HOME}/output/vin500pos.txt", "w+") as f:
        # write elements of list
        for items in pos:
            f.write("%s\n" % items)
        print("File written successfully")
    # close the file
    f.close()
