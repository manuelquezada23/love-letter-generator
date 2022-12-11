## Coding Cupids - Love Letter Generator

by Manuel Quezada (mquezad1), Samantha Gundotra (sgundotr), Jose Urruticoechea (jurrutic), Juan Pablo Ramos Barroso (jramosba)

### How to run
1) Download this [poetry_data](https://drive.google.com/file/d/1ln5h7Kavsbkl5aDKYz3-iRUgBc3X5DnU/view) folder
2) Create a `data` folder in the top-level directory
3) Create a `processed` folder in the `data` folder and add a file called `processed_poems.pickle`
4) Add the `poetry_data` folder to the `data` folder
5) Download this [models](https://drive.google.com/drive/folders/1EErNArFD6KRrlJaJyANgeURUZdqHLTh-?usp=sharing) folder and add it to the `code` directory
6) Run `python love_letter.py`

### Introduction
For our project we want to build a love-letter generator to revive modern romance. Users will input their recipient and we will generate a love letter (chunk of text) tailored to them. Our project falls into Natural Language Processing (NLP).

### Related Work
[How I Build an AI Poetry Generator](https://medium.com/voice-tech-podcast/how-i-build-an-ai-poetry-generator-1254f7335c17)

This paper uses OpenAI’s GPT-2 pretrained language model to train their own text prediction model. To turn it into a poet, the author fine-tuned the model by feeding it a database of modern poems. 

[GPT-2 Simple](https://github.com/minimaxir/gpt-2-simple)

This GitHub repository is a simple Python package that wraps existing model fine-tuning and generation scripts for OpenAI's GPT-2 text generation model (specifically the "small" 124M and "medium" 355M hyperparameter versions). Additionally, this package allows easier generation of text, generating to a file for easy curation, allowing for prefixes to force the text to start with a given phrase.

### Data
We plan on combining a dataset of love letters and text from romantic books from different time periods to incorporate old-fashioned english. Some of our proposed sources include: 

- [Collection of Love Letters — Kaggle](https://www.kaggle.com/datasets/fillerink/love-letters)
- [Poems Dataset — Kaggle](https://www.kaggle.com/datasets/michaelarman/poemsdataset)
- [Collection of Famous Love Letters — The Romantic](https://theromantic.com/LoveLetters/main.htm)
- Potentially use love letters from books / world wide web.

### Methodology

Our model’s architecture will be similar to assignment 4. It will involve RNN variants and transformers that will allow the model to produce sentences that make sense in the English language and are love themed.

### Metrics

##### What experiments do you plan to run?
Have a test group of users and send each of them a generated letter
Ask them to fill out a Google Form about their thoughts (will come up with survey questions)
 
##### For most of our assignments, we have looked at the accuracy of the model. Does the notion of “accuracy” apply for your project, or is some other metric more appropriate?
We won’t be working with labeled data so the notion of accuracy is not applicable to our project. However, we can measure how realistic the letters generated are by using other metrics such as perplexity. 
 
##### If you are doing something new, explain how you will assess your model’s performance.
- Perplexity
- Survey Ratings
#####  What are your base, target, and stretch goals?
Base Goal: Generate a grammatically correct love letter.
Target Goal: Generate an emotionally charged, moving, and grammatically correct love letter.
Stretch Goal: Receive input from a user regarding their personality, information on the receiver of the letter, and submit some of their own writing to inform the semantics and style of the love letter — from which a grammatically correct and personalized love letter would be made.
 
#####  Potential metrics:
- Perplexity
- Cross entropy
-Bits-per-character

### Ethics
##### Why is Deep Learning a good approach to this problem?

Deep Learning is a good approach to this problem because it will allow us to easily interpret large amounts of data and form them into meaningful information. It will allow us to create human-like written love letters that will satisfy the needs and interests of our users. It can also give users inspiration and expose them to alternative styles of letter writing.

##### Who are the major “stakeholders” in this problem, and what are the consequences of mistakes made by your algorithm?
 
The major “stakeholders” are the users that will be utilizing this program to create love letters. They are the ones who, for various reasons, will want to use a love letter generator powered by Deep Learning. Some of the consequences of mistakes made by our algorithm aren’t really that serious; however, they can impact the experience of a user by providing a faulty or bad love letter.
 
### Division of Labor
The work would be distributed as follows:
- Manuel Quezada: model architecture + training and accuracy

- Samantha Gundotra: Model architecture + data collection + preprocessing

- Jose Urruticoechea: Model architecture + training and accuracy

- Juan Pablo Ramos Barroso: Model architecture + data collection + preprocessing

If we have time: everyone frontend (website to display love letters)
