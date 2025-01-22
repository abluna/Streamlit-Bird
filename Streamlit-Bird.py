import pandas as pd
import numpy as np
import streamlit as st
import stbrd
from PIL import Image
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing import image
from keras.applications.resnet_v2 import preprocess_input, decode_predictions

st.write("""
# Birds of San Diego
The goal of this tool is to quickly predict a bird species based on a single image. \n
To run a prediction, upload a photo and click "Predict"
""")

######################
## Import the model ##
######################

@st.cache_resource
def load_model():

    bird_model = keras.saving.load_model("hf://abluna/bird_classification_v4")
    
    return bird_model

#########################
## Importing the image ##
#########################

img = st.file_uploader("Upload the image", type=None)

left_co,cent_co,last_co = st.columns(3)
with cent_co:   
    if img is not None:
        original_image = Image.open(img)
        st.image(original_image, caption="Your Image", width = 250)
 
###########################
## Importing Keras Model ##
###########################


bird_index_list = stbrd.create_bird_index_list()
bird_link_df = stbrd.create_bird_image_links()

click_predict_message = "Predict Bird Species"

if img is not None:
    if st.button(click_predict_message):
        with st.spinner("Wait for it..."):

            # Use the function to load your data
            bird_model = load_model()

            tf_model = bird_model
            index_list = bird_index_list
            targ_size = 800

             # `img` is a PIL image of size 224x224
            img_v2 = image.load_img(img, target_size=(targ_size, targ_size))

            # `x` is a float32 Numpy array of shape (300, 300, 3)
            x = image.img_to_array(img_v2)

            # We add a dimension to transform our array into a "batch"
            # of size (1, 300, 300, 3)
            x = np.expand_dims(x, axis=0)

            # Finally we preprocess the batch
            # (this does channel-wise color normalization)
            x = preprocess_input(x)

            preds = tf_model.predict(x)

            ## Get list of predictions
            pred_dict = dict(zip(index_list, np.round(preds[0]*100,2)))
            Sorted_Prediction_Dictionary = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)

            Count_5Perc = preds[0][preds[0]>0.02]

            if len(Count_5Perc) == 1:
                TopPredictions = Sorted_Prediction_Dictionary[0]
                to_df = list(TopPredictions)
                df = pd.DataFrame({"Species": to_df[0], "Probability":to_df[1]}, index=[0])
            if len(Count_5Perc) > 1:
                TopPredictions = Sorted_Prediction_Dictionary[0:len(Count_5Perc)]
                df = pd.DataFrame(TopPredictions, columns =["Species", "Probability"])

            df["Probability"] = df["Probability"].round(2)
            df = df.merge(bird_link_df, how="left", on="Species")
            df['Species'] = df['Species'].str.slice(0, -5)

            # st.dataframe(
            #     df,
            #     column_config={
            #         "name": "App name",
            #         "Probability": st.column_config.NumberColumn(
            #             "Probability",
            #             format="%.2f%%"),
            #         "Link": st.column_config.ImageColumn("Image", width='small')
            #     },
            #     hide_index=True
            # )
            with cent_co:
                st.image(list(df["Link"]), caption = list(df["Species"]), width = 200)



            