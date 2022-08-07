import pandas as pd
import numpy as np
import re
import pickle

from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


#load data
url = "https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv"
df_raw = pd.read_csv(url)

#drop duplicates
df_raw = df_raw.drop_duplicates().reset_index(drop = True)

#fuction to remove stopwords
stopWord = ['is','you','your','and', 'the', 'to', 'from', 'or', 'I', 'for', 'do', 'get', 'not', 'here', 'in', 'im', 'have', 'on',
're', 'https', 'com', 'of']  

def remove_stopwords(urlData):
  if urlData is not None:
    words = urlData.strip().split()
    words_filtered = []
    for word in words:
      if word not in stopWord:
        words_filtered.append(word)
    result = " ".join(words_filtered) #hace un join elemento por elemento separados por espacio
  else:
      result = None
  return result

df['url'] = df['url'].apply(remove_stopwords)


# varias funciones

def comas(text):
    """
    Elimina comas del texto
    """
    return re.sub(',', ' ', text)

def espacios(text):
    """
    Elimina enters dobles por un solo enter
    """
    return re.sub(r'(\n{2,})','\n', text)

def minuscula(text):
    """
    Cambia mayusculas a minusculas
    """
    return text.lower()

def numeros(text):
    """
    Sustituye los numeros
    """
    return re.sub('([\d]+)', ' ', text)

def caracteres_no_alfanumericos(text):
    """
    Sustituye caracteres raros, no digitos y letras
    Ej. hola 'pepito' como le va? -> hola pepito como le va
    """
    return re.sub("(\\W)+"," ",text)

def comillas(text):
    """
    Sustituye comillas por un espacio
    Ej. hola 'pepito' como le va? -> hola pepito como le va?
    """
    return re.sub("'"," ", text)

def palabras_repetidas(text):
    """
    Sustituye palabras repetidas

    Ej. hola hola, como les va? a a ustedes -> hola, como les va? a ustedes
    """
    return re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

def esp_multiple(text):
    """
    Sustituye los espacios dobles entre palabras
    """
    return re.sub(' +', ' ',text)


# funcón para eliminar https
def url(text):
    return re.sub(r'(https://www|https://)', '', text)

df['url_limpia'] = df['url'].apply(url).apply(caracteres_no_alfanumericos).apply(esp_multiple)

df['is_spam'] = df['is_spam'].apply(lambda x: 1 if x == True else 0)

#Usar técnicas de NLP para preprocesamiento de datos
vec = CountVectorizer().fit_transform(df['url_limpia'])

def clean_data(urlData):
  
    #remove punctuation, digit, simbols
    urlData = re.sub('[^a-zA-Z]', ' ', urlData)
    
    #duplicate space
    urlData = re.sub(r'\s+', ' ',  urlData)
    #urlData=" ".join(urlData.split())

    urlData = re.sub(r'\b[a-zA-Z]\b', ' ',urlData)  #\b word boundary

    urlData = urlData.strip()   #remove space on right and left include tab
    return urlData

df['url'] = df['url'].str.lower() 
#clean-data
df['url'] = df['url'].apply(clean_data)
# se limpia url
df['url_limpia'] = df['url'].apply(url).apply(caracteres_no_alfanumericos).apply(esp_multiple)








