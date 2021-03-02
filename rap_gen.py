import pickle
import streamlit as st
import rg_functions as dlf
from PIL import Image

#### READING IDX_V_CHAR w/FTS ####
with open('Pickles/idx_v_char_fts.pickle', 'rb') as f:
    idx_v_char_fts = pickle.load(f)
    f.close()
    
#### READING CHAR_V_IDX w/FTS ####
with open('Pickles/char_v_idx_fts.pickle', 'rb') as f:
    char_v_idx_fts = pickle.load(f)
    f.close()

#### READING CLEANED LYRICS w/FTS ####
with open('Pickles/cleaned_lyrics_string_fts.pickle', 'rb') as f:
    cleaned_lyrics_string_fts = pickle.load(f)
    f.close()

st.set_page_config(page_title='Rap Generator 2000', layout='wide')
st.markdown("""
<style>
body {
    color: #303030;
    background-color: #f2f2f2;
}
</style>
""", unsafe_allow_html=True)

"""
# Rap Generator 2000
###### *An artificial neural network trained to generate lyrics by Darius Fuller*
------

### **_Have you_**: 
* Ever wondered if a computer could rap like a person?
* Need help finishing your song, but no one around to help?
* Want to laugh at some quirky computer-generated sentences?

If you've said yes, then try using the *Rap Generator 2000*! (RG2K)

## Instructions

1. Adjust the parameters in the sidebar
2. Place some words in the box (*no punctuation*)
3. Click that button!
"""

with st.sidebar.header('1. How much help do you need?'):
    num2gen = st.sidebar.slider('Number of characters to generate', 0, 400, 200, 20)

with st.sidebar.header('2. How complex are the lyrics you need'):
    temperature = st.sidebar.slider('High numbers lead to more "creative" generation', 1, 100, 75, 1)

## Create tensorflow model
model_gen = dlf.build_app_model(cleaned_lyrics_string_fts)

## Create TextGenerator object
text_generator = dlf.TextGenerator(model_gen,
                                   idx_v_char_fts,
                                   char_v_idx_fts)

## Taking input
user_input = st.text_input('Input lyrics, then press ENTER')

if st.button('Generate lyrics!'):

    ## Create new text based upon input string + store it
    text_generator.generate_text(user_input, num2gen, (temperature/100))
    new_lyrics = text_generator.get_generated_text()

    ## Censor object
    censor = dlf.Censor(lyrics=new_lyrics)

    ## Input modification
    censor.str_splitter()
    censor.execute_censoring()
    censor.str_joiner('M')

    ## Censored output
    new_output = censor.get_lyrics('J')

    st.markdown(f'Characters generated: {num2gen}')
    st.markdown(f'Complexity level: {temperature}%')

    st.markdown('**New lyrics:**')
    st.text(new_output)

"""
### Tips & Notes
* The Rap Generator 2000 (RG2K) is best used with English input, as it only outputs English lyrics
* If the RG2K gets stuck in a loop repeating the same phrase or word, re-submit your input text and it should generate a different set of lyrics.
* Having been trained on explicit lyrics, I have censored the common offensive words within the RG2K's vocabulary. However, the phrases it comes
up with may still refer to vulgar topics.
* You can enter as many words as needed to give the RG2K a start, but it works best with a single line or <10 words.
* The RG2K is in the early stages with respect to performance. As a result, it is not consistent at producing a proper rhyme scheme. 
Time permitting, I intend add in lyrics from other artists to further experiment with outputs and improve lyric quality. For this first
iteration, I wanted to see how well it could perform on a small scale.

------
## Background and Summary

This project was inspired by a love for music and a great [video](https://www.youtube.com/watch?v=ZMudJXhsUpY) by Laurence Moroney demonstrating
how to generate text using Tensorflow.

The RG2K uses a Tensorflow classification neural network trained on 180 rap songs collected from
the internet. I selected this sample of songs from one rap artist's collection, hoping to see if the 
extreme bias would produce a computer "clone" of the rapper.

I chose this rapper due in part to their relatively consistent output and accessibility of song lyrics. 
On the other hand, I wanted to see how their repetitive/catchy, flashy style translates through the 
networks predictions.

*Can you guess who it is?*
"""

if st.button('Reveal rapper'):
    ## Show the truth
    """
    ## **TYGA**
    """
    tyga = Image.open('Images/dom-hill-0wMLZNbE8Ac-unsplash_2.jpg')

    st.image(tyga, use_column_width=True)
    st.markdown('Photo credit: [Dom Hill](https://unsplash.com/@ohthehumanity?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText) *via Unsplash*')
    
"""
### Useful Links:
[GitHub](https://github.com/dfuller22/)

[Medium Page](https://dariuslfuller.medium.com/) *Blog(s) coming soon*

[LinkedIn](https://www.linkedin.com/in/darius-fuller/)

[Twitter](https://twitter.com/dariuslfuller)

"""