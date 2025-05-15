import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

DATA_SET_PATH = "./letter+recognition/letter-recognition.data"
OUTPUT_DIR = "./Plots"
SAVE = False

column_names = [
    "letter", "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar",
    "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"
]


df = pd.read_csv(DATA_SET_PATH, names=column_names)

plt.figure(figsize=(12, 5))
sns.countplot(x=df["letter"], order=sorted(df["letter"].unique()))
plt.title("Letters distribution in data set")

if SAVE:
    plt.savefig(os.path.join(OUTPUT_DIR, "Letters distribution in data set.png"), dpi=300, bbox_inches='tight')
    plt.close()
else:
    plt.show()
