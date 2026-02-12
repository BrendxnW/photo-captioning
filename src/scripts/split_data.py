import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/captions.csv")

images = df["image"].unique()

train_img, temp_img = train_test_split(images, test_size=0.2, random_state=42)
test_img, val_img = train_test_split(temp_img, test_size=0.5, random_state=42)

train_df = df[df["image"].isin(train_img)]
test_df = df[df["image"].isin(test_img)]
val_df = df[df["image"].isin(val_img)]

train_df.to_csv("src/data/Train/train.csv", index=False)
test_df.to_csv("src/data/Test/test.csv", index=False)
val_df.to_csv("src/data/Validate/validate.csv", index=False)