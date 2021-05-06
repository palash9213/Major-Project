import streamlit as st
from bs4 import BeautifulSoup
import requests
import pandas as pd
import contractions
import re
import emoji
from PIL import Image
import nltk
nltk.download('stopwords')
stopwordList = nltk.corpus.stopwords.words('english')

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vs = SentimentIntensityAnalyzer()

from nltk.tokenize import ToktokTokenizer
tokenizer = ToktokTokenizer()


def build_dataframe(url):
  review_list = []
  try:
    page=requests.get(url)
    soup=BeautifulSoup(page.content,'html.parser')
    review_titles=soup.findAll('a',attrs={'class':'title'})   #list of review title
    for title in review_titles:
      review_list.append(title.get_text()[:-1])
    #print(review_list)
    return review_list
    
  except:
    st.error("Error: Enter the movie name correctly")


def get_review_url(movie_name):
  html_page = requests.get("https://www.imdb.com/find?q="+movie_name+"&ref_=nv_sr_sm")
  soup = BeautifulSoup(html_page.content)
  dic={}


  for link in soup.find_all('a'):
    n_link=link.get('href')
    try:
      if n_link.startswith('/title/') and link.get_text() is not '' :
        dic[link.get_text()]='https://www.imdb.com/'+n_link+'reviews?ref_=tt_ql_3'
    except:
      pass

  st.subheader("Copy the exact name of the movie from the search list of movies\n")
  for key,values in dic.items():
    st.write(key)
  url_key=st.text_input(" ")
  try:
    url=dic[url_key]
    data=build_dataframe(url)
    return data
  except:
    st.warning("Warning: Enter the movie name correctly")


def removeStopwords(text):
      tokens = tokenizer.tokenize(text)
      tokens = [token.strip() for token in tokens]
      filtered_tokens = [token for token in tokens if token not in stopwordList]
      filteredText = '  '.join(filtered_tokens)
      return filteredText

#REMOVING HTML TAGS

def html_tag(text):
  soup = BeautifulSoup(text, "html.parser")
  new_text = soup.get_text()
  return new_text

#EXPANDING THE WORDS WRITTEN IN SHORT FORM

def con(text):
  expand = contractions.fix(text)
  return expand
    
#REMOVING SPECIAL CHARACTERS

def removeSpecialCharacters(text):
  pattern = r'[^A-Za-z0-9\s]'
  text = re.sub(pattern, '', text)
  return text

def toEmoji(val):
  review = []
      
  if val <= -0.05:
    review.append(emoji.emojize(":disappointed_face:"))
  elif -0.05 < val < 0.05:
    review.append(emoji.emojize(":neutral_face:"))
  else:
    review.append(emoji.emojize(":smiling_face_with_smiling_eyes:"))
  return review

def finalEmoji(Review):
  emojis = []
  for i in Review:
    emojis.append(i[0])
  return emojis

def preprocess(data):
  df = pd.DataFrame(data, columns=["Reviews"])
  stopwordList.remove('no')
  stopwordList.remove('not')
    
  #APPLYING ALL PREPROCESSING STEPS ON OUR DATASET

  df.Reviews = df.Reviews.apply(lambda x:x.lower())


  df.Reviews = df.Reviews.apply(html_tag)


  df.Reviews = df.Reviews.apply(con)


  df.Reviews = df.Reviews.apply(removeSpecialCharacters)


  df.Reviews = df.Reviews.apply(removeStopwords)

  df['compound'] = df['Reviews'].apply(lambda x: vs.polarity_scores(x)['compound'])
    
  #CREATING FINAL SENTIMENT ANALYSIS DATAFRAME BASED ON COMPOUND SCORE

  emoji_column = df['compound'].apply(toEmoji)
  emoji_final = finalEmoji(emoji_column)
  emojis_column = pd.Series(emoji_final, name="EMOJIS")
  dfs = [df, emojis_column]
  result = pd.concat(dfs, axis=1)

  final_data = result.drop(['compound'], axis = 1)
  return final_data

st.title("MOVIE REVIEWS SENTIMENTAL ANALYSIS")
st.sidebar.header("Meaning of the Emoji's")

img = Image.open("images/positive_img.jpg")
st.sidebar.image(img, width=150)
st.sidebar.write("This is the Emoji for positive review ")

img = Image.open("images/neutral_img.png")
st.sidebar.image(img, width=150)
st.sidebar.write("This is the Emoji for neutral review ")

img = Image.open("images/negative_img.jpg")
st.sidebar.image(img, width=150)
st.sidebar.write("This is the Emoji for negative review ")

st.header("Enter the movie name ")
movie_name=st.text_input("")
if movie_name is not None:
  data = get_review_url(movie_name)
if data is not None:
  final_data=preprocess(data)
  st.write(final_data.head())
  if st.checkbox('Show full reviews'):
    final_data
