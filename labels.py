import pandas as pd

df = pd.read_csv(r"D:\DuyguTespiti/labels.csv")

label_map = {
    'anger': 0,
    'contempt': 1,
    'disgust': 2,
    'fear': 3,
    'happy': 4,
    'neutral': 5,
    'sad': 6,
    'surprise': 7
}

df['label'] = df['label'].map(label_map)
df.to_csv(r"D:\DuyguTespiti/labels_numeric.csv", index=False)
