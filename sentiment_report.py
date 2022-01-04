import streamlit as st 
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px


@st.cache()
def convert_df(df):
     return df.to_csv().encode('utf-8')


st.write("""
# Interaktiv sentiment analyse
""")

st.sidebar.image('https://willandagency.com/wp-content/uploads/2019/05/willandagency_logo_white_final.svg', width=250)

with st.sidebar.subheader('Upload fil'):
    uploaded_file = st.sidebar.file_uploader("Upload CSV med sentiment labels", type=["csv"])
st.sidebar.subheader("Noter")
text_contents = st.sidebar.text_area(label="Tag evt. noter her", value="Skriv noter her", height=30)
st.sidebar.download_button(
	label="Download noter",
    data=text_contents,
    file_name='noter.txt',
    mime='text/csv',
)
st.sidebar.subheader("")


if uploaded_file is not None:
  	dataframe = pd.read_csv(uploaded_file, sep = ",")


dataframe.columns = [c.replace(' ', '_') for c in dataframe.columns]


#dataframe = dataframe.rename(columns={ dataframe.columns[20]: "new_col_name" })


# ---------------- INTERACTIVE DATAFRAME MODULE ---------------

emotion_list = dataframe.Sentiment.dropna().unique()
date_list = dataframe.Alternate_Date_Format.dropna().unique()
source_list = dataframe.Source.dropna().unique()


emotion_option = st.multiselect('Vælg sentiment', emotion_list)
date_option = st.multiselect('Vælg dato', date_list)
source_option = st.multiselect('Vælg kilde', source_list)


emotion_filter = dataframe['Sentiment'].isin(emotion_option)
date_filter = dataframe['Alternate_Date_Format'].isin(date_option)
source_filter = dataframe['Source'].isin(source_option)

interactive_df = dataframe[emotion_filter & date_filter & source_filter] 


st.write(interactive_df[['clean_text', 'Sentiment', 'URL', 'Source', 'Alternate_Date_Format', 'Reach', 'City', 'State', 'Subregion']])


#Download csv  

csv = convert_df(interactive_df)

#Front end download of csv 
st.download_button(
	label="Download filtered CSV",
    data=csv,
    file_name='filtered_sentiment_classification.csv',
    mime='text/csv',
)



#---------------- LINE CHART MODULE ------------------------

line_df = pd.DataFrame({'count' : dataframe.groupby( [ "Alternate_Date_Format", "Sentiment"] ).size()}).reset_index()
line_fig = px.line(line_df, x="Alternate_Date_Format", y="count", color='Sentiment')
line_fig.update_xaxes(autorange="reversed")

st.plotly_chart(line_fig, use_container_width =True)

with st.expander("Sortér efter sentiment"):
	line_list = dataframe.Sentiment.dropna().unique()
	line_option = st.multiselect('Vælg', line_list)
	line_emotion_filter = line_df['Sentiment'].isin(line_option)



	line_df1 = line_df[line_emotion_filter]
	line_fig1 = px.line(line_df1, x="Alternate_Date_Format", y="count", color='Sentiment', labels=dict(count="Antal sætninger", Sentiment="Sentiment", Alternate_Date_Format="Dato"))
	line_fig1.update_xaxes(autorange="reversed")

	st.plotly_chart(line_fig1, use_container_width = True)



#--------WORDCLOUDS TEST --------------------------------

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import numpy as np
import matplotlib.pyplot as plt
#from PIL import Image


def cloud(text, max_word, max_font, random):
    stopwords = ["ad",
"af","aldrig","alene","alle","allerede","alligevel","alt","altid","anden","andet",
"andre","at","bag","bare","begge","bl.a.","blandt","blev","blive","bliver","burde","bør","ca.","da","de","dem","den","denne","dens","der",
"derefter","deres","derfor","derfra","deri","dermed","derpå","derved","det","dette","dig","din","dine","disse","dit","dog","du","efter","egen","ej","eller","ellers",
"en","end","endnu","ene","eneste","enhver","ens","enten","er","et","f.eks.","far","fem","fik","fire","flere","flest","fleste","for","foran","fordi","forrige",
"fra","fx","få","får","før","først","gennem","gjorde","gjort","god","godt","gør","gøre","gørende","ham","han","hans","har","havde","have","hej","hel",
"heller","helt","hen","hende","hendes","henover","her","herefter","heri","hermed","herpå","hos","hun","hvad","hvem","hver","hvilke","hvilken","hvilkes","hvis","hvor","hvordan",
"hvorefter","hvorfor","hvorfra","hvorhen","hvori","hvorimod","hvornår","hvorved","i","igen","igennem","ikke","imellem","imens","imod","ind","indtil","ingen",
"intet","ja","jeg","jer","jeres","jo","kan","kom","komme","kommer","kun","kunne","lad","langs","lav","lave","lavet","lidt","lige","ligesom","lille",
"længere","man","mand","mange","med","meget","mellem","men","mens","mere","mest","mig","min","mindre","mindst","mine","mit","mod","må","måske","ned",
"nej","nemlig","ni","nogen","nogensinde","noget","nogle","nok","nu","ny","nyt","når","nær","næste","næsten","og","også","okay","om","omkring","op",
"os","otte","over","overalt","pga.","på","RT","samme","sammen","se","seks","selv","selvom","senere","ser","ses","siden","sig","sige","sin","sine",
"sit","skal","skulle","som","stadig","stor","store","synes","syntes","syv","så","sådan","således","tag","tage","temmelig","thi","ti","tidligere","til",
"tilbage","tit","to","tre","ud","uden","udover","under","undtagen","var","ved","vi","via","vil","ville","vor","vore""vores","vær","være","været","øvrigt"]
    
    wc = WordCloud(background_color="white", colormap="hot", max_words=max_word,
    stopwords=stopwords, max_font_size=max_font, random_state=random)

    wc2 = WordCloud(background_color="white", colormap="cool", max_words=max_word,
    stopwords=stopwords, max_font_size=max_font, random_state=random)

    # generate word cloud
    wc.generate(text)

    wc2.generate(text2)



    # show the figure
    fig = plt.figure()


    fig, axes = plt.subplots(1,2, gridspec_kw={'width_ratios': [1, 1]})
    axes[0].imshow(wc, interpolation="bilinear")
    axes[1].imshow(wc2, interpolation="bilinear")
  

    for ax in axes:
        ax.set_axis_off()
    st.pyplot(fig)



st.write("## Sentiment Summarization with wordclouds")

st.sidebar.write("Settings for WordCloud")
max_word = st.sidebar.slider("Max words", 200, 3000, 200)
max_font = st.sidebar.slider("Max Font Size", 50, 350, 60)
random = st.sidebar.slider("Random State", 30, 100, 42 )


#Generate text for wordcloud

vrede_df = dataframe.loc[dataframe["Sentiment"] == "Vrede"]
vrede_text = vrede_df["clean_text"].str.cat(sep=" ")

glæde_df = dataframe.loc[dataframe["Sentiment"] == "Glæde"]
glæde_text = glæde_df["clean_text"].str.cat(sep=" ")

tristhed_df = dataframe.loc[dataframe["Sentiment"] == "Tristhed"]
tristhed_text = tristhed_df["clean_text"].str.cat(sep=" ")

foragt_df = dataframe.loc[dataframe["Sentiment"] == "Foragt"]
foragt_text = foragt_df["clean_text"].str.cat(sep=" ")

overraskelse_df = dataframe.loc[dataframe["Sentiment"] == "Overraskelse"]
overraskelse_text = overraskelse_df["clean_text"].str.cat(sep=" ")

frygt_df = dataframe.loc[dataframe["Sentiment"] == "Frygt"]
frygt_text = frygt_df["clean_text"].str.cat(sep=" ")

cloudchoice = [f"VREDE:{vrede_text}", f"GLÆDE:{glæde_text}", f"TRISTHED:{tristhed_text}",
 f"FORAGT:{foragt_text}", f"OVERRASKELSE:{overraskelse_text}", f"FRYGT:{frygt_text}"]

text = st.selectbox("Vælg tekst", cloudchoice)
text2 = st.selectbox("Vælg tekst 2", cloudchoice)
if text is not None:
	if st.button("Plot"):
		st.write("### Word cloud")
		st.write(cloud(text, max_word, max_font, random), use_column_width=True)



#--------------- MODULE 1 --------------------------
st.subheader('Sentiment for all data')
#Create sentiment plot for total count of emotions
sentiment_values = pd.DataFrame(dataframe['Sentiment'].value_counts())
sentiment_values.index.name = 'emotion'
sentiment_values.reset_index(inplace=True)

fig = px.pie(sentiment_values, values='Sentiment', names='emotion', title = 'Total fordeling af sentiment i procent')

st.plotly_chart(fig, use_container_width=True)

with st.expander("Se sætninger"):
	emotion_list = dataframe.Sentiment.dropna().unique()
	emotion_option = st.selectbox('Vælg sentiment:', emotion_list)
	emotion_df = dataframe.loc[dataframe['Sentiment'] == emotion_option]
	st.write(emotion_df[['clean_text', 'URL', 'Source', 'Date', 'Reach']])


#---------------- MODULE 1.5 -----------------------


#--------------- MODULE 2 --------------------------
st.subheader('Sentiment by keyword')

keywords = dataframe.Keywords.dropna().unique()
keyword_option = st.selectbox('Select keyword', keywords) 

keyword_df = dataframe.loc[dataframe['Keywords'] == keyword_option]

#create plot
sentiment_values_keyword = pd.DataFrame(keyword_df['Sentiment'].value_counts())
sentiment_values_keyword.index.name = 'emotion'
sentiment_values_keyword.reset_index(inplace=True)

fig_kw = px.pie(sentiment_values_keyword, values='Sentiment', names='emotion', title = f'Fordeling af sentiment i procent for keyword: {keyword_option}, ')

st.plotly_chart(fig_kw, use_container_width=True)


#--------------- MODULE 3 --------------------------

st.subheader('Sentiment by source')

sources = dataframe.Source.dropna().unique()
source_option = st.selectbox('Select source', sources) 

source_df = dataframe.loc[dataframe['Source'] == source_option]

sentiment_values_source = pd.DataFrame(source_df['Sentiment'].value_counts())
sentiment_values_source.index.name = 'emotion'
sentiment_values_source.reset_index(inplace=True)

fig_s = px.pie(sentiment_values_source, values='Sentiment', names='emotion', title = f'Fordeling af sentiment i procent for kilde: {source_option}')

st.plotly_chart(fig_s, use_container_width=True)



