# Sentiment Analisys on Movie Reviews dataset
Implementation of Sentiment Analysis through *Subjectivity Detection* and *Polarity Classification* using a structure seeing involved a Naive Bayes Classifier jointly with a Support Vector Machine (SVM). \
The work is inspired by the original paper [A Two-Stage Classifier for Sentiment Analysis, Nguyen and Pham](https://aclanthology.org/I13-1114.pdf).

## Install Dependencies
A Conda Environment is provided within the repository to make the dependencies installation easier. To install it please run the following command from the repository root:

```bash
    conda env create --file nlu-environment.yml
```

All dependencies should be included in the environment, however we might still miss out on some dependencies from `nltk`. A forthright solution to such an issue is to run the following commands after the Conda Environment is set to leave:
```bash
    python
    nltk.download()
```

## Run the Code
First off open the [Preprocessing Notebook](01-preprocessing.ipynb) and run all cells. If you prefer to run everything directly from terminal, run the following from reposotory root:
```bash
    jupyter nbconvert --to notebook --inplace --execute 01-preprocessing.ipynb
```

Then, to run the main code run the following command from repository root:
```bash
    python 02-main.py
```

