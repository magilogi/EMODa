#Loading dependencies
import streamlit as st
import pandas as pd
import re
import seaborn as sns
import io
from simpletransformers.language_representation import RepresentationModel
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification



#Defining functions

def clean(text):
    text = re.sub("@[A-Za-z0-9]+","",text) #Remove @ sign
    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text) #Remove http links
    text = " ".join(text.split())
    text = text.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
    return text


def remove_emojis(text):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', text)


def clean_json(sentence): #crude function to extract label from json output of huggingface pipeline
  sentence = re.sub(r'[^a-zA-Z æøå]+', '', sentence)
  sentence = sentence.split()
  sentence = sentence[1]
  return sentence


#@st.cache() #Caching doesnt work with this model for some reason
#def load_model():
	#model_path = "NikolajMunch/danish-emotion-classification"
	#classifier = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
#	return(classifier)


@st.cache()
def classify(text):
	Bert_E = text.apply(classifier)
	return(Bert_E)

#------------------------- setup for download

def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

#--------------------------
@st.cache()
def load_embeddings_model():
	model = RepresentationModel(
    	model_type="bert",
        model_name='Maltehb/danish-bert-botxo',
        use_cuda=False)
	return(model)

@st.cache()
def get_embeddings(sentences):
  return model.encode_sentences(sentences,combine_strategy="mean")



@st.cache
def convert_df_tsv_output(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv(sep='\t', index=None, header=None)


@st.cache
def convert_df_tsv_meta(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv(index=False, sep='\t')


#--------------------------------------

#Script

showWarningOnDirectExecution = True

#---FRONT END -----------

st.write("""
# File generator for sentiment analysis and Embeddings Projector
""")

st.write("""
Upload CSV file here
""")

#Upload widget
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  	dataframe = pd.read_csv(uploaded_file, sep = "\t", encoding = "UTF-16")

#What type of sentiment is to be analyzed? Twitter / Headlines
option = st.selectbox(
	'Select column to analyze',
	('Body', 'Headline'))  	

st.write('Column selected:', option)

RT_option = st.selectbox(
	'Delete retweets and duplicate comments?',
	('True', 'False'))

st.write('Delete duplicates set to:', RT_option)

#---------------Pre-processing data input-------------------


dataframe = dataframe[0:100] # FOR TEST PURPOSES

if RT_option == True:
		dataframe = dataframe.drop_duplicates(subset=option, keep="first")


dataframe = dataframe[dataframe[option].notna()] #only show rows with text in 'option'
dataframe = dataframe.drop(['Sentiment'], axis=1)
	

clean_text = dataframe[option].map(lambda x: clean(x))
clean_text = clean_text.map(lambda x: remove_emojis(x))
clean_text = clean_text.to_frame()

clean_text = clean_text[clean_text[option].str.split().str.len().lt(200)] #Temporary fix for overflowing tokens error


#Brug igen når streamlit kan cache funktion der loader model
#Loading classification model
#classifier = load_model() 

#Classifying sentences
#Bert_E = classify(clean_text)

model_path = "NikolajMunch/danish-emotion-classification"
#tokenizer = AutoTokenizer.from_pretrained('NikolajMunch/danish-emotion-classification')


classifier = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

Bert_E = clean_text[option].apply(classifier)

#Assigning the values to df and merging with original data

clean_text = clean_text.assign(Emotion=Bert_E.values) #Assigning classification result to sentences
clean_text['Emotion'] = clean_text['Emotion'].str.get(0) #Json cleaning pt 1
clean_text['Emotion'] = clean_text['Emotion'].astype(str) #Making it a string

E_clean = clean_text["Emotion"].apply(clean_json) #Extracting only the emotion label
clean_text = clean_text.assign(Emotion=E_clean.values) #Assigning cleaned values


clean_text = clean_text.rename(columns={"Body": "clean_text", "Emotion": "Sentiment"})



df_full = pd.concat([dataframe, clean_text], axis=1, ignore_index=False)
csv = convert_df(df_full)



#Front end download of csv 
st.download_button(
	label="Download data as CSV",
    data=csv,
    file_name='sentiment_classification.csv',
    mime='text/csv',
)


st.write(" Genererer TSV filer til Embedding Projector - Vent venligst...")



#------Generating embeddings and metadata containing the 

metadata = df_full[['Sentiment', 'clean_text', 'URL', 'Keywords']]

metadata['clean_text'] = metadata['clean_text'].astype(str)
metadata['Sentiment'] = metadata['Sentiment'].astype(str)
metadata['Keywords'] = metadata['Keywords'].astype(str)

model = load_embeddings_model()

e = get_embeddings(metadata["clean_text"])

# Convert NumPy array of embedding into data frame
embedding_df = pd.DataFrame(e)

output = convert_df_tsv_output(embedding_df)

st.download_button(
	label="Download output",
    data=output,
    file_name='output.tsv',
    mime='text/csv'
)


meta_df = convert_df_tsv_meta(metadata)



st.download_button(
	label="Download metadata",
    data=meta_df,
    file_name='metadata.tsv',
    mime='text/csv'
)

link = '[Embedding Projector](https://projector.tensorflow.org/)'
st.markdown(link, unsafe_allow_html=True)

