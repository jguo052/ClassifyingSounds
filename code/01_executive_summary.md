# Executive Summary

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

## Summary of Notebooks