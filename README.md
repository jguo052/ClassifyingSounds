# ClassifyingSounds

---

## Problem Statement
#### Motivation
Over the last decade, many companies have incorporated automatic speech recognition into their technology. iPhone users all have access to Siri, which listens to speech and tries to understand what action it should perform. Zoom and Youtube both offer speech-to-text options. Many more applications can be found [here](https://en.wikipedia.org/wiki/Speech_recognition).

For anyone who has used a device that performs speech recognition, they have likely seen the device make numerous errors. Below is an example of a Google Voice error [found on the internet](https://www.technologizer.com/2010/08/22/worst-google-voice-transcription-errors/). Automatic speech recognition is challenging. 
> hi allen my name is white and my number is area code (626) 523-8023 once again the number is (562) 652-3808

There are many things that make speech recognition difficult, one of which is that people frequently make sounds that are not words to be parsed. For example, a person may cough or clear their throat and some speech recognition software would either have to identify it as such, or ignore it altogether and only pick out the spoken words.

#### Problem Statement
Our goal is to train a neural network that takes in short audio files (less than six seconds) and classifies the file as either a cough, laugh, sigh, sneeze, sniff, or throat clear. We will deem our model successful if it achieves an accuracy score of 90% or higher on unseen data.

The hope with this project is that our model could eventually be used alongside a speech recognition model and pick out sounds that the speech recognition model does not need to attempt to parse.

---

## The Dataset
You can find the dataset used for this project [here](https://github.com/YuanGongND/vocalsound#Download-VocalSound). It consists of 21,024 short audio recordings, all of which are labeled as either a cough, laugh, sigh, sneeze, sniff, or throatclearing. These samples were provided by 3,365 different individuals who all contributed at least one file for each kind of sound.

During exploration, it was not difficult to find examples of audio files that were either mislabelled or completely unrelated to any of the labels. It is worth noting that this likely decreases the potential for reaching our desired model metric of 90% accuracy.

This dataset also came with some information about the different speakers like gender, age, country, and language. Although we do not use this speaker information to train the model, it was a valuable resource for assessing our production model's performance.


---

## Following Along
To obtain the same production model, one should run the files in the `code` directory in numerical order. The order is:
- `01_audio_examples.ipynb`
- `02_extracting_features.ipynb`
- `03_mfcc_cnn.ipynb`
- `04_melspec_cnn.ipynb`
- `05_model_analysis.ipynb`

Note that files `03` and `04` were run on [Kaggle](https://www.kaggle.com/) to make use of their public GPU to train the convolutional neural networks (CNN). This means to follow along with these two notebooks, you will need to update the relative paths of any files read into or saved from `03` and `04`.

Lastly, there is a file that generates a *Streamlit* application in the main directory titled `streamlit.py`. If you have the Streamlit library installed, you can access this app by navigating to this project's directory on your local machine in the terminal and then entering `streamlit run streamlit.py`. This will open up an interactive page where you can try out the model by creating your own short recordings.

Beyond the most typical data science libraries, what you will need to follow along are `librosa` for extracting audio features, `tensorflow` for training neural networks, and `streamlit` for interacting with a live version of the model.

---

## Summary

One way to create a classification model on short audio files is by converting the files into images that contain the most relevant audio information. Then those images can be passed through convolutional neural networks to perform image classification. This is the approach we take in this project.

First we needed to make sure the audio data were all of the same size. This guarantees that the image data we extract from these audio files are also the same size and can be fed into the neural network without any issue. This was accomplished by extracting the audio signal from each file using `librosa` and then trimming and padding that signal to six seconds.

The two kinds of images we extract from our audio files are called *Mel Frequency Cepstral Coefficients (MFCCs)* and *Mel Spectrograms*. These images generally look like heatmaps and an example for each is shown below.

![MFCCs]('images/cough_mfccs.jpg')

![Mel Spectrogram]('images/cough_melspec.jpg')



We created two neural networks, one for each kind of image. Both models performed fairly similarly and the accuracy scores on the validation sets generally stuck between 85% and 87% from epoch to epoch. The MFCCs model seemed a little more reliable at the end of it all and produced more consistent results across the different classes. This can be demonstrated by the confusion matrices below. 

![MFCCs confusion matrix]('images/melspec_confusion_matric.jpg')

![Mel Spectrogram]('images/melspec_confusion_matrix.jpg')

We see that the first matrix corresponding to the MFCCs data has the most trouble with distinguishing between coughs and throat clears, but besides that, the numbers off the diagonal are generally low. On the other hand, the values off the diagonal are consistently higher for the Mel Spectrogram confusion matrix.

With our MFFCs model we generated predictions on the test set and analyzed the results. For this analysis on the test set we focused on two things:

- speaker data that was provided alongside the audio files
- prediction confidence given by the prediction probabilities

The main takeaways were that the information about the speaker (gender, age, language) had little relationship to how well the model performed. This was expected, but worth looking into as gender, age, and language do have influence on how a person sounds. As for prediction confidence, we noticed that the model more misclassified coughs with notably higher levels of confidence than the other sounds. It performed quite well on laughters and sighs and the confidence levels for misclassifying these sounds tended to be lower.

![Average Confidence]('images/avg_conf_preds.jpg')

---

## Conclusions

Unfortunately, we did not meet our goal of achieving an accuracy score above 90% for model. We were however able to get close by achieving an accuracy score of 87%. Although this is not above our original goal, it seems fair given the numerous mislabelled and erroneous audio files in the dataset.

During our exploration of our model we noticed that it could be most improved better distinguishing coughs and throat clears. This is not too surprising since these classes do sound very similar.

In addition to this, we developed a streamlit application that gives others the opportunity to toy around with the model. In this app you can record yourself making one of the six sounds (within six seconds) and submit this recording into the model to generate a prediction. The page then produces the prediction, a bar chart of its confidence levels for each class, and then visuals of your recording (as a waveform, MFCCs, and Mel Spectrogram).

---

## Next Steps

One natural next step is to collect more data on other kinds of sounds like (yawns, grunts, etc). This would make the model more applicable for predicting more general non-speech human sounds.

Another next step along the same lines as above is to collect more data on coughs and throat clears. Our model had the most difficulty deciding which of these two categories a sound might be. Having more examples for the model (especially accurately labeled ones) could help it identify these two classes more precisely.

The last recommendation for progressing this project is to find a way to connect it to speech recognition. If this project be incorporated into a speech recognition model, it could possibly strengthen its performance by dealing with sounds the speech recognition system does not need to make an effort to understand.