# Voice Recognition Project using Hidden Markovian Models And Recursive Neural Networks
Voice Recognition Project by Aidinis Athanasios, Charalampos Kastoris.
----------------------------------------------------------------------
>The purpose is to implement a speech processing and recognition system for isolated word recognition. In the first step, appropriate acoustic features are extracted from voice data using Python packages, which are then analyzed and mapped to understand and extract useful information. In essence, these features are a collection of cepstral coefficients that are obtained after the signals are examined using a specially created filter bank that was motivated by psychoacoustic studies.
----------------------------------------------------------------------
* For steps 1-7 run -> main.py
* For step 8 run -> step8.py
* or steps 9-13 run -> main_2.py
* For step 14 run -> main_3.py
----------------------------------------------------------------------
Important notes:
* Files to be read (parsed) must be in the same directory with the project
* All the scripts in this file must be on the same file 
(scripts like lib.py etc are imported to main scripts)
----------------------------------------------------------------------
>Details:
* custom_NB_classifier.py -> Contains the NB clf created in lab 1 and some helper functions
* hmmgmm.py -> Contains the HMM GMM model and its train,predict and gridsearch functions we created
* lib.py -> Contains useful functions used in this project and all the RNN models 
(used in steps 8 and 14)
* optimising_clfs.py -> Contains functions that gridsearch and optimize sklearn clfs
* parser.py -> Helper code to parse the data 
* pytorchtools.py -> Early stopping class from https://github.com/Bjarten/early-stopping-pytorch


