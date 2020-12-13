# Documentation

In the following, we will describe how the text classification model is build and its major components.
## Project Overview
In this project, we are focusing on text classification, sarcasm detection in specific. Given a specific Twitter response, we are trying to tell if the response is sarcasm. Each data point in the dataset consist of the response tweet and also the context of the response. The goal of this project is to develop a robust model that can perform well in telling whether the given Twitter response is sarcasm or not.

## DEMO Link
* https://mediaspace.illinois.edu/media/1_pa3h5d8y

## Classification Model
We have experimented with many different models. In the end, we reached the competition benchmark score by fine-tuning a pre-trained BERT, [**distilled BERT**](https://huggingface.co/transformers/model_doc/distilbert.html) in specific.

### Data Preprocessing
For data cleaning, we remove punctuations and any other special characters in the Twitter response. Also, we expand abbreviations such as can't and won't into can not and will not respectively. We also remove the heading of each response. After cleaning the data, we use a pre-trained [**DistilBertTokenizer**](https://huggingface.co/transformers/model_doc/distilbert.html#distilberttokenizer) to tokenize the cleaned response data. Then, the tokenized responses are used as the input to our model.

### Model Architecture
The general idea of our model is to fine-tune the pre-trained distilled BERT for text classification. We achieved this by adding two extra fully connected linear layers and fixing the parameters for the BERT model. The general pipeline of the model is that given tokenized responses as input, we first put the inputs to the distilled BERT base model to get high-dimensional representations of the responses. Then the response representations are input to the two linear layers to get the final prediction of whether the response is sarcasm. Between the linear layers, we used [**ReLU**](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html) as activation. We also applied dropout to both the output of the base model and the output of the linear layers.

### Model Training
Given raw Twitter response, we first preprocessing it following the data preprocessing steps to get tokenized responses. Then, the tokenized responses are used as the input to the BERT base model, which will give high-dimensional representations of the responses. Then, those representations are put into several linear layers to generate the final prediction. The loss function we use is a [**NLL loss**](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html). After getting the training prediction, together with the ground truth labels that indicate whether the response is sarcasm or not, we put them to the NLLLoss and performs back propagation on the computed loss.


### Evaluation
During training, we split the train dataset into two subset, one for actual training, one for validation. The percentage of the validation set is 20% of the data point in the original training dataset. In each epoch, we evaluate the F1 score of the model on the validation set save the model having the best F1 score. For the actual prediction task on the test set, we use the saved model during training for the actual prediction.

## Previous Attempts
We have come a long way to the model we have right now. We first thought of models based on CNN and RNN. But after actually implemented them, those models did not give results good enough to beat the competition benchmark. Apart from distilled BERT, we have also experimented with the full BERT, which gives decent result, but it tends to overfit and takes a lot more time to run. We have also tried to vary the number of linear layers and the dimension of those layers used to fine-tune the model, we have tried to add 3 or 4 linear layers and many other different combinations of dimension, but we finalized to 2 linear layers, which are of size(768, 256) and (256, 2). In terms of different activation function, we have tried [**Tanh**](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html), [**PReLU**](https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html#torch.nn.PReLU), and [**LeakyReLU**](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html). Though they all give very similar results, we choose ReLU in the end. We have also experimented the dropout ratio in the range [0, 0.5]. We observed that with 0.5 dropout, the model reaches best performance on the test set.

For data preprocessing, we found that removing stopwords and stemming the words has negatively affected the performance of our model. Expanding abbreviations seems to have improve the performance of the model by reducing overfitting. Removing punctuations and special characters generally gives cleaner data for the tokenizer. Therefore, it helps both with the model training and model testing.


## Dependencies
* Python
* Json
* PyTorch
* Skit-Learn
* Transformers


To install dependencies, you can use the included **environment.ysml** to create a virtual environment with Anaconda. Installation reference can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). A detailed tutorial will be included in the DEMO.

## Contributions
* Junting Wang: Team Leader. Implemented the model and written up the code documentation.
* Tianwei Zhang: Helped with model testing and project DEMO.


