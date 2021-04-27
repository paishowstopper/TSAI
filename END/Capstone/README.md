END Capstone Project
====================

This project is to generate python code when provided an input sentence (intention) in simple English.

Data preparation, extension and cleaning
========================================

Apart from the dataset provided in the course, added conala dataset which consisted of close to 2900 more examples of text-code pairs (in json format). The original dataset questions were first loaded to a list (# was the natural symbol to differentiate questions from answers). After that, loaded the training and test json files from Conala dataset (http://www.phontron.com/download/conala-corpus-v1.1.zip) and then extracted the questions (intent in json) and answers (snippet in json) and loaded them to the initial list (Note: Prefixed the intent with #). The complete list was then processed to extract the questions and answers to separate lists. The questions were extracted to "questions" list (by checking for the # prefix) and the answers to "answers" list (by iterating through the lines till we find the next #, i.e., next question). Each answer was processed to add a space between alphabets/numbers and possible special characters leading/trailing them, e.g., add(a,b) becomes add ( a , b ). This is important to tokenize all the python code words correctly.

Also, considered loading the 100 END assignment 8 text-code pairs but finally decided not to (Did not want the model to know the questions I'll be using for final testing).

All the necessary datasets were initially loaded to google drive and then downloaded from there.

Tokenizer
=========

Used the spacy tokenizer since it was giving expected results.

Pre-trained embeddings
======================

The output without any pre-trained embeddings was bad. The answers were not related to the intended tasks at all. Tried 2 pre-trained embeddings. The first one did not improve the output but using Word2Vec significantly improved the results. Word2Vec is commonly used in NLP to learn word associations from a large corpus of text. This pre-training can detect synonymous words or suggest additional words for partial lines. The embeddings were performed on the combination of source and target and then added back to the source and target separately (based on the check that they were present in these earlier).

NOTE: Set the minimum frequency to 1 for target (Python code) to ensure none of the code tokens are missed (our overall dataset is not very large).

Model
=====

The model used was mostly the same as the one used in the "Attention is all you need" class except for the use of pre-trained token embeddings in encoder and decoder. The model parameters used were also same as in the class (Was not able to experiment a lot with the parameters because of the CUDA out of memory. Even when ). Even the batch size had to be reduced to 16. Tried to customize the positional embedding layer but did not give good results.

Loss function
=============

Cross-entropy loss function gave very poor output (generated code was terrible). As suggested by a student in the END Telegram group, used label-smoothing which gave excellent results. Label smoothing is a regularization technique for classification problems to prevent the model from predicting the labels too confidently during training and generalizing poorly. Label smoothing replaces one-hot encoded label vector y_hot with a mixture of y_hot and the uniform distribution (https://towardsdatascience.com/what-is-label-smoothing-108debd7ef06). Got the loss calculator from here (https://github.com/jadore801120/attention-is-all-you-need-pytorch). Used a smoothing value of 0.1.

Training was conducted for 50 epochs because the losses were saturating at around this mark.

25 examples and the corresponding attention graphs have been shown in the colab file (last section)

Additional points
=================

1. Still feel the model submitted is not giving great results. If we use the same intent as in the trained dataset, the outputs are good but anything significantly different or not present in the dataset is giving weird results. Not sure if this is because the dataset is small (and because of CUDA out of memory errors, was not able to train it more).
2. There are unnecessary spaces between the code. e.g.:- "num = int ( input ( " Enter a number : ") ) ". When spaces are important, the code is not working correctly.
