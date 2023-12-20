import pandas as pd

music_data = pd.read_csv('./data/music.csv')
x = music_data.drop(columns=['genre', 'age'])
print(music_data)
print(x)
