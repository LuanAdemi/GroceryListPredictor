# The Model
The model currently consists out of the following sub-models:

## PClassifier

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LuanAdemi/GroceryListPredictor/master?filepath=pclassifier%2FPClassifier.ipynb)

This sub-model is capable of classifying a large amount of products just by their name. We used a bunch of shopping receipts to create a solid data basis for the training. The model consists of a simple, yet powerful NLP approach with RNN's, which enables the network to recognize certain word stems and making predictions based on the order of letters.

Moreover, we plan on using a feedback system for expanding our dataset with (hopefully) every shopping trip. The user will be able to act as a critic and can correct the predictions of the classifier. Hence, the model will improve with the time and amount of users.

## RScanner

(Binder link coming soon)

This sub-module is responsable for retrieving the items from a reciept, the user uploaded to the webapp. There is currently no heavy development around this sub-model. We will add a short explanation soon.

## PRecommender

(Binder link coming soon)

This sub-module uses the information gained by the other sub-modules to predict the items you should buy next time you are going to a supermarket. There too is no heavy development around this sub-module at the moment.



For full documentations for every sub-model in an interactive environment, see the binder links above. 
