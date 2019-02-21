import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 케글에서 사용된 영화 데이터셋 2만개를 학습시켜 사용자가 선택한 영화와 비슷한 영화를 찾아주는 시스템.
data = pd.read_csv('./movies_metadata.csv', low_memory=False)
data = data.head(20000)

data['overview'] = data['overview'].fillna('')

# overview의 문서 내용을 벡터화 시킴.
vectorizer = TfidfVectorizer(stop_words='english')
x = vectorizer.fit_transform(data['overview'])
tfidf_dict = vectorizer.get_feature_names()

data_array = x.toarray()
df = pd.DataFrame(data_array, columns=tfidf_dict)

indices = pd.Series(data.index, index=data['title']).drop_duplicates()

# 사용자가 선택한 영화.
user_title = 'Toy Story'
index = indices[user_title]

# 사용자가 선택한 영화와 전체 문서와의 코사인 유사도 구하기
df = df.dot(df.iloc[index])

df = pd.DataFrame({'index':df.index, 'similarities':df.values})
df = df.sort_values(by='similarities', ascending=False)

# 상위 10개의 데이터만 뽑음
df = df[1:11]

lst = df.index
print(data['title'].iloc[lst])
