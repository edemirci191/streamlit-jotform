import numpy as np
import pandas as pd
import streamlit as st
import annoy
import tensorflow_hub as hub
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from langdetect import detect
import requests
import os
import pickle
import nltk
from urllib.request import urlopen
from tensorflow_text import SentencepieceTokenizer

@st.cache(allow_output_mutation=True, ttl=120000, max_entries=1)
def load_lda_model():
  #before this model 200_pass20 we used
  lda_model = pickle.load(urlopen("https://storage.googleapis.com/jotform-recommender.appspot.com/lda_model_tfidf_clean_v2.pkl"))
  return lda_model

@st.cache(allow_output_mutation=True, ttl=120000, max_entries=1)
def nltk_download():
  nltk.download('wordnet')
  
@st.cache(allow_output_mutation=True, ttl=120000, max_entries=1)
def lemmatize_stemming(text):
  nltk_download()
  return WordNetLemmatizer().lemmatize(text, pos='v')

@st.cache(allow_output_mutation=True, ttl=120000, max_entries=1)
def preprocess(text):
  result = []
  for token in gensim.utils.simple_preprocess(text):
    if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
      result.append(lemmatize_stemming(token))
  return result

@st.cache(allow_output_mutation=True, ttl=120000, max_entries=1)
def topic_recommend(user_input):
  topic_result=[]
  lda_model = load_lda_model()
  dictionary = load_dictionary()
  bow_vector = dictionary.doc2bow(preprocess(user_input))
  for index,score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    if score > 0.1:
      for i in range(1):
        if(lda_model.show_topic(index, 1)[i][0] not in topic_result):
          if(lda_model.show_topic(index, 1)[i][0] != "thank"):
            topic_result.append(lda_model.show_topic(index, 1)[i][0])
          break
  return topic_result

def apply_url(id):
  full_url = "https://www.jotform.com/answers/" + str(id)
  return full_url

@st.cache(allow_output_mutation=True, ttl=120000, max_entries=1)
def load_map():
       #this was mapping.pkl before
       mapobj = pickle.load(urlopen("https://storage.googleapis.com/jotform-recommender.appspot.com/mapping_clean.pkl"))
       return mapobj

@st.cache(allow_output_mutation=True, ttl=120000, max_entries=1)
def load_question():
       #this was question.pkl before
       queobj = pickle.load(urlopen("https://storage.googleapis.com/jotform-recommender.appspot.com/questions_clean.pkl"))
       return queobj

@st.cache(allow_output_mutation=True, ttl=120000, max_entries=1)
def load_matrixes():
       #this was matrix.pkl before
       matrixobj = pickle.load(urlopen("https://storage.googleapis.com/jotform-recommender.appspot.com/matrix_clean.pkl"))
       return matrixobj

@st.cache(allow_output_mutation=True, ttl=120000, max_entries=1)
def load_wfilter():
       wfilterobj = pickle.load(urlopen("https://storage.googleapis.com/jotform-recommender.appspot.com/badwords.pkl"))
       return wfilterobj
       
@st.cache(allow_output_mutation=True, ttl=120000, max_entries=1)
def load_dictionary():
       #this was just dictionary.pkl before
       dictionary = pickle.load(urlopen("https://storage.googleapis.com/jotform-recommender.appspot.com/dictionary_clean_v2.pkl"))
       return dictionary

@st.cache(allow_output_mutation=True, ttl=120000, max_entries=1)
def load_stemmer():
       english_stemmer = pickle.load(urlopen("https://storage.googleapis.com/jotform-recommender.appspot.com/english_stemmer.pkl"))
       return english_stemmer

def find_similar_items(lang_index,mapping_name,embedding, num_matches=5):
  '''Finds similar items to a given embedding in the ANN index'''
  ids = lang_index.get_nns_by_vector(
  embedding, num_matches, search_k=-1, include_distances=False)
  items = [mapping_name[i] for i in ids]
  return items,ids

embedding_dimension = 64
@st.cache(allow_output_mutation=True, ttl=120000, max_entries=1)
def load_model():
       model_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'
       embeda = hub.load(model_url)
       return embeda

def extract_embeddings(query,embed_fn,rpm): 
  '''Generates the embedding for the query'''
  query_embedding =  embed_fn([query])[0].numpy()
  if rpm is not None:
    query_embedding = query_embedding.dot(rpm)
  return query_embedding


base_url = "https://www.jotform.com/answers/"

def main():
  converted_list = load_wfilter()
  random_projection_matrix_en, random_projection_matrix_de, random_projection_matrix_tr, random_projection_matrix_pt, random_projection_matrix_it,random_projection_matrix_es,random_projection_matrix_nl,random_projection_matrix_fr = load_matrixes()
  question_en, question_de, question_tr,question_pt, question_it, question_es, question_nl, question_fr =  load_question()
  mapping_en, mapping_de, mapping_tr, mapping_pt, mapping_it, mapping_es, mapping_nl, mapping_fr = load_map()
  embed = load_model()
  st.sidebar.image("https://storage.googleapis.com/jotform-recommender.appspot.com/jotform-logo-transparent-800x400.png",width=300)
  menu = ["Similar Questions","Relevant Topics"]
  choice = st.sidebar.selectbox("Menu",menu)
  if choice == "Similar Questions":
    st.title("Jotform Support Forum Question Recommender")
    st.subheader("Overview")
    st.write("Purpose of this application is to recommend the user similar questions that has been asked before by other users. When the user asks a new question other already answered similar questions are going to be recommended to the user in English and also in his/her native language.")
    uinput = st.text_input("Question","How can I create a succesful survey form ?")    
    user_input = st.text_area("Description","I want to learn how to create succesful survey form")
    st.button("Search Question")
    allinput = uinput +" "+ user_input
    user_input = allinput
    st.image('https://storage.googleapis.com/jotform-recommender.appspot.com/podo_7.png',width=264)
    not_found = 1
    # in else part embedfn=embed_.. changed to embedfn = embed
    if any(word in user_input for word in converted_list):
      print('Your sentence contains profanity words, please try again')
    else:
      l_index = None
      map = None
      embedfn = None
      rpm = None
      lg = detect(user_input)
      varforid = None

      if lg == 'en':
        if not os.path.exists('en_from_url'):
          #this was index_en before
          url_en = "https://storage.googleapis.com/jotform-recommender.appspot.com/index_en_clean"
          r_en = requests.get(url_en, stream = True)
          with open("en_from_url","wb") as f:
            for block in r_en.iter_content(chunk_size = 8192):
              if block:
                f.write(block)
        index_filename_en = "en_from_url"
        index_en = annoy.AnnoyIndex(embedding_dimension)
        index_en.load(index_filename_en, prefault=True)
        l_index = index_en
        map = mapping_en
        embedfn = embed
        rpm = random_projection_matrix_en
        not_found = 0
        varforid = question_en

      if lg == 'es':
        if not os.path.exists('es_from_url'):
          url_es = "https://storage.googleapis.com/jotform-recommender.appspot.com/index_es"
          r_es = requests.get(url_es, stream = True)
          with open("es_from_url","wb") as f:
            for block in r_es.iter_content(chunk_size = 8192):
              if block:
                f.write(block)
        index_filename_es = "es_from_url"
        index_es = annoy.AnnoyIndex(embedding_dimension)
        index_es.load(index_filename_es, prefault=True)
        l_index = index_es
        map = mapping_es
        embedfn = embed
        rpm = random_projection_matrix_es
        not_found = 0
        varforid = question_es

      if lg == 'tr':
        if not os.path.exists('tr_from_url'):
          url_tr = "https://storage.googleapis.com/jotform-recommender.appspot.com/index_tr"
          r_tr = requests.get(url_tr, stream = True)
          with open("tr_from_url","wb") as f:
            for block in r_tr.iter_content(chunk_size = 8192):
              if block:
                f.write(block)
        index_filename_tr = "tr_from_url"
        index_tr = annoy.AnnoyIndex(embedding_dimension)
        index_tr.load(index_filename_tr, prefault=True)
        l_index = index_tr
        map = mapping_tr
        embedfn = embed
        rpm = random_projection_matrix_tr
        not_found = 0
        varforid = question_tr

      if lg == 'fr':
        if not os.path.exists('fr_from_url'):
          url_fr = "https://storage.googleapis.com/jotform-recommender.appspot.com/index_fr"
          r_fr = requests.get(url_fr, stream = True)
          with open("fr_from_url","wb") as f:
            for block in r_fr.iter_content(chunk_size = 8192):
              if block:
                f.write(block)
        index_filename_fr = "fr_from_url"
        index_fr = annoy.AnnoyIndex(embedding_dimension)
        index_fr.load(index_filename_fr, prefault=True)
        l_index = index_fr
        map = mapping_fr
        embedfn = embed
        rpm = random_projection_matrix_fr
        not_found = 0
        varforid=question_fr

      if lg == 'de':
        if not os.path.exists('de_from_url'):
          url_de = "https://storage.googleapis.com/jotform-recommender.appspot.com/index_de"
          r_de = requests.get(url_de, stream = True)
          with open("de_from_url","wb") as f:
            for block in r_de.iter_content(chunk_size = 8192):
              if block:
                f.write(block)
        index_filename_de = "de_from_url"
        index_de = annoy.AnnoyIndex(embedding_dimension)
        index_de.load(index_filename_de, prefault=True)
        l_index = index_de
        map = mapping_de
        embedfn = embed
        rpm = random_projection_matrix_de
        not_found = 0
        varforid = question_de

      if lg == 'nl':
        if not os.path.exists('nl_from_url'):
          url_nl = "https://storage.googleapis.com/jotform-recommender.appspot.com/index_nl"
          r_nl = requests.get(url_nl, stream = True)
          with open("nl_from_url","wb") as f:
            for block in r_nl.iter_content(chunk_size = 8192):
              if block:
                f.write(block)
        index_filename_nl = "nl_from_url"
        index_nl = annoy.AnnoyIndex(embedding_dimension)
        index_nl.load(index_filename_nl, prefault=True)
        l_index = index_nl
        map = mapping_nl
        embedfn = embed
        rpm = random_projection_matrix_nl
        not_found = 0
        varforid= question_nl

      if lg == 'pt':
        if not os.path.exists('pt_from_url'):
          url_pt = "https://storage.googleapis.com/jotform-recommender.appspot.com/index_pt"
          r_pt = requests.get(url_pt, stream = True)
          with open("pt_from_url","wb") as f:
            for block in r_pt.iter_content(chunk_size = 8192):
              if block:
                f.write(block)
        index_filename_pt = "pt_from_url"
        index_pt = annoy.AnnoyIndex(embedding_dimension)
        index_pt.load(index_filename_pt, prefault=True)
        l_index = index_pt
        map = mapping_pt
        embedfn = embed
        rpm = random_projection_matrix_pt
        not_found = 0
        varforid = question_pt

      if lg == 'it':
        if not os.path.exists('it_from_url'):
          url_it = "https://storage.googleapis.com/jotform-recommender.appspot.com/index_it"
          r_it = requests.get(url_it, stream = True)
          with open("it_from_url","wb") as f:
            for block in r_it.iter_content(chunk_size = 8192):
              if block:
                f.write(block)
        index_filename_it = "it_from_url"
        index_it = annoy.AnnoyIndex(embedding_dimension)
        index_it.load(index_filename_it, prefault=True)
        l_index = index_it
        map = mapping_it
        embedfn = embed
        rpm = random_projection_matrix_it
        not_found = 0
        varforid = question_it

      if not_found == 1:
        if not os.path.exists('en_from_url'):
          #this was index_en before
          url_en = "https://storage.googleapis.com/jotform-recommender.appspot.com/index_en_clean"
          r_en = requests.get(url_en, stream = True)
          with open("en_from_url","wb") as f:
            for block in r_en.iter_content(chunk_size = 8192):
              if block:
                f.write(block)
        index_filename_en = "en_from_url"
        index_en = annoy.AnnoyIndex(embedding_dimension)
        index_en.load(index_filename_en, prefault=True)
        l_index = index_en
        map = mapping_en
        embedfn = embed
        rpm = random_projection_matrix_en
        lg = 'en'
        varforid = question_en

      query_embedding = extract_embeddings(user_input,embedfn,rpm)
      items,ids = find_similar_items(l_index,map,query_embedding, 5)
      lst=[]
      show_df = pd.DataFrame()
      if lg != 'en': 
        index_filename_en = "en_from_url"
        index_en = annoy.AnnoyIndex(embedding_dimension)
        index_en.load(index_filename_en, prefault=True)
        query_embedding_en = extract_embeddings(user_input,embedfn,random_projection_matrix_en)
        items_en,ids_en = find_similar_items(index_en,mapping_en,query_embedding_en, 5)
        extended_items = items + items_en
        for j in ids_en:
          lst.append("https://www.jotform.com/answers/"+str(question_en.iloc[j]['id']))
        for i in ids:
          lst.append("https://www.jotform.com/answers/"+str(varforid.iloc[i]['id']))
        show_df.to_html(escape=False)
        show_df['Similar Questions'] = extended_items
        show_df['Thread URL'] = lst
        #st.subheader("Most Related Topics for")
        #topic = topic_recommend(extended_items[0])
        #st.write(topic)
        st.subheader('Recommendations')
        st.table(show_df)
        #st.write("[https://www.jotform.com/answers/]" + str(lst[1])) hyperlink with constant id
      else:
        for i in ids:
          lst.append("https://www.jotform.com/answers/" + str(varforid.iloc[i]['id']))

        show_df.to_html(escape=False)
        #st.write("here")
        #st.write(lg)
        #st.write(detect(user_input))
        show_df['Similar Questions'] = items
        show_df['Thread URL'] = lst
        #st.subheader("Most Related Topics for")
        #st.write(user_input)
        #topic = topic_recommend(user_input)
        #st.write(topic)
        st.subheader('Recommendations')
        st.table(show_df)
        
  if choice == "Relevant Topics":
    st.write("Relevant Topics")
    uinput = st.text_input("Question","How can I create a succesful survey form ?")    
    user_input = st.text_area("Description","I want to learn how to create succesful survey form")
    st.button("See Relevant Topics")
    allinput = uinput +" "+ user_input
    user_input = allinput
    lg = detect(user_input)
    if lg == 'en':
      topic = topic_recommend(user_input)
      st.write(topic)
    else:
      index_filename_en = "en_from_url"
      index_en = annoy.AnnoyIndex(embedding_dimension)
      index_en.load(index_filename_en, prefault=True)
      embedfn=embed
      query_embedding_en = extract_embeddings(user_input,embedfn,random_projection_matrix_en)
      items_en,ids_en = find_similar_items(index_en,mapping_en,query_embedding_en, 5)
      topic = topic_recommend(items_en[0])
      st.write(topic)
if __name__ == '__main__':
  main()
