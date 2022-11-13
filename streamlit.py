# basic imports
import numpy as np
import matplotlib.pyplot as plt

# audio processing imports
import sounddevice as sd
import soundfile as sf
import librosa, librosa.display

# load pretrained CNN
from tensorflow.keras.models import load_model

# dashboard
import streamlit as st


# create containers
headerblock = st.container()
recordblock = st.container()
confidenceblock = st.container()
visualblock = st.container()


# constants
DURATION = 6
SR = 22050
labels = ['cough', 'laugh', 'sigh', 'sneeze', 'sniff', 'throat clear']



def record(duration = DURATION, sr = SR):
    # records the user for DURATION seconds with a sample rate of SR
    sd.default.samplerate = sr
    sd.default.channels = 1
    
    # with st.spinner() provides a timer icon and listed text while recording
    with st.spinner('recording... you have six seconds to make the noise of your choice'):
        recording = sd.rec(int(duration * sr))
        sd.wait(duration)
    
    # store recording as a wav file
    # this seems simplest for displaying the recording back to the user
    sf.write('recording.wav', recording, samplerate = SR, subtype = 'PCM_24')
    
    return recording


def extract_mfccs(signal, sr = SR):
    # from a 1-d array and given sample rate, generate MFCCs
    # using 20 coefficients
    mfcc = librosa.feature.mfcc(y = signal, n_mfcc = 20, n_fft = 2048, hop_length = 512)
    
    # the CNN model excepts 4 dimensions (num of recordings, mfccs, frames, channels)
    # there is only 1 recording and channel, so apend axes to front and back
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    
    return mfcc


def extract_melspec(signal, sr = SR):
    # from a 1-d array and given sample rate, generate mel-spectrogram
    # using 128 mel filterbanks
    melspec = librosa.feature.melspectrogram(y = signal,
                                             n_mels = 128,
                                             n_fft = 2048,
                                             hop_length = 512)
    
    # convert melspec to dB scale
    log_melspec = librosa.power_to_db(melspec)
    
    # the CNN model excepts 4 dimensions (num of recordings, mels, frames, channels)
    # there is only 1 recording and channel, so apend axes to front and back
    log_melspec = log_melspec[np.newaxis, ..., np.newaxis]
    
    return log_melspec



with headerblock:
    st.title('Classifying Non-Speech Sounds')
    
    # brief header description of project
    st.write("We built a model which **classifies short audio files** (six seconds or shorter) \
             as either a **cough, laugh, sigh, sneeze, sniff, \
             or throat clear**. Below you can try this model for yourself! \
             There is a button to record yourself for six \
             seconds, and another button that feeds this audio data into the model and \
             generates a prediction.")
    
    # use columns to center header image
    col1, col2, col3 = st.columns([0.5,2,0.5])
    col2.image('images/mainart.png', caption=None, width=400, use_column_width=False, clamp=False, channels="RGB", output_format="auto")

    # mention of visuals at bottom
    st.write('At the bottom of the page you can also see \
             images of features that were extracted from your recording and \
             used to generate the predictions (waveform, MFCCs, mel-spectrogram).')
             
    st.write('---')



with recordblock:
    st.header("Record Yourself")
    
    # describe what user should do to try out the model
    st.write("Click the **record** button below to begin recording yourself for six seconds. \
             Make one of the sounds (cough, laugh, sigh, sneeze, \
            sniff, throat clear) and we will try and guess which one it is!")       

    # include a record button  =  
    if st.button('Record'):
        # if this is not the first time recording (and predictions have been made)
        # clear the predictions and visuals from the screen
        if 'probs' in st.session_state:
            st.session_state.pop('probs')
        
        # recording is a 1-d array presented as a 2-d
        # flatten so we can convert to MFCC or Mel Spectrogram
        recording = record().flatten()

        # saves recording if other page buttons are clicked
        st.session_state['audio'] = recording


    # if a recording exists, display the audio for playback
    if 'audio' in st.session_state:
        st.audio('recording.wav')
        st.write('If you want to create a different recording, click the **record** button again.')
        
        st.write('---')

        
        st.header("Here's what we think...")
        st.write('Click on the **predict** button below to generate our predictions for the sound clip.')
        
        # text gets overwritten - 'Thinking...' gets replaced with 'That sounds like...'
        with st.empty():
            if st.button('Predict'):
                # brief loading visual labeled with 'Thinking...'
                with st.spinner('Thinking...'):
                    # load pretrained CNN model (keras)
                    model = load_model('model.h5')
                    
                    # extract MFCCs from stored audio signal
                    mfcc = extract_mfccs(st.session_state['audio'])
                
                    # compute the CNNs expected probability for each sound
                    probs = model.predict(mfcc)
                    
                    # select the highest probability sound label
                    # and print to user
                    class_num = np.argmax(probs)
                    pred = labels[class_num]
                st.write(f'*That sounds like a {pred} to me!*')
                
                # store label (as index and word) and probabilities
                st.session_state['pred_index'] = class_num
                st.session_state['pred'] = pred
                st.session_state['probs'] = probs[0].round(2)
        
    

# if predictions have been generated
# display confidence as bar chart, waveform, MFFCs, and Mel Spectrogram
if 'probs' in st.session_state and 'pred_index' in st.session_state:
    with confidenceblock:
        
        # create matplotlib bar chart of prediction probabilities
        fig, ax = plt.subplots(figsize = (8,4))
        
        # make predicted sound's bar dark blue and all others light blue
        colors = ['lightsteelblue' for i in range(6)]
        colors[st.session_state['pred_index']] = 'tab:blue'
        
        fig.suptitle('Our Confidence In The Prediction', size = 15)
        ax.bar(x = labels, height = st.session_state['probs'], color = colors, width = 0.6)
        
        ax.set_ylabel('Confidence', size = 10)
        
        # only use quartile tick marks for simplicity
        ax.set_yticks([0,0.25, 0.5, 0.75, 1])
        # have upper y-lim of 1.1 so there is buffer space for confident predictions
        ax.set_ylim([0,1.1])
        # remove tick marks
        ax.tick_params(axis = 'both', bottom = False, left = False)
        
        ax.grid(axis = 'y', alpha = 0.5)
        
        # plot image
        st.pyplot(fig)
        
        st.write('---')
        
    
    with visualblock:
        st.header('Visuals of Your Recording')
        
        # add whitespace below header
        st.write('#')
        
        # extract MFCCs and Mel Spectrograms for visuals
        # shapes are (1,a,b,1) for feeding into model
        # only need to middle two numbers (a,b) for displaying as image
        signal = st.session_state['audio']
        mfcc = extract_mfccs(signal).reshape(20,259)            # a=20, b=259
        mel_spec = extract_melspec(signal).reshape(128,259)     # a=128, b=259
        
        # create a 3x1 grid of images (waveform, MFCCs, and Mel Spectrogram)
        fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize = (8, 12))
        
        # generate waveform image, x-axis labeled in seconds
        im0 = librosa.display.waveshow(signal, 
                                       x_axis = 's', 
                                       ax = ax[0])
        
        ax[0].set_title('Waveform', size = 15)
        ax[0].set_xlim([0,6])   # six seconds
        ax[0].set_xticks([0,1,2,3,4,5,6])
        ax[0].set_ylabel('Amplitude', size = 10)
        # remove y-axis tick marks
        ax[0].tick_params(axis = 'y', labelleft = False, left = False)


        # generate MFCCs image, x-axis labeled in seconds
        im1 = librosa.display.specshow(mfcc,
                                       x_axis = 's',
                                       ax = ax[1])
        
        ax[1].set_title('MFCCs', size = 15)
        ax[1].set_ylabel('MFCCs', size = 10)


        # generate Mel Spectrogram image, x-axis labeled in seconds
        im2 = librosa.display.specshow(mel_spec, x_axis = 's', ax = ax[2])
        ax[2].set_title('Mel-Spectrogram', size = 15)
        ax[2].set_ylabel('Mels', size = 10)
     
        # set the vertical spacing between subplots
        plt.subplots_adjust(hspace=0.6)

        # plot 3x1 grid of images
        st.pyplot(fig)
        