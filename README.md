# BioTokens

This is a **Transformer** based neural machine translation model.

## Data

The dataset is generated using our inhouse amino-acid potential analytical model

## Data Process

Data is generated automatically when the model is run. Data generation solves the forward problem of amino acid potential fingerprints, whereas the ML model attempts to solve the inverse problem.

### Word Segmentation

- **Tool**：[sentencepiece](https://github.com/google/sentencepiece)
- **Preprocess**：Run `./data/get_corpus.py` , in which we will get bilingual data to build our training, dev and testing set.  The data will be saved in `corpus.en` and `corpus.ch`, with one sentence in each line.
- **Word segmentation model training**: Run `./tokenizer/tokenize.py`, in which the *sentencepiece.SentencePieceTrainer.Train()* mothed is called to train our word segmentation model. After training, `ami.model`，`ami.vocab`，`meas.model` and `meas.vocab` will be saved in `./tokenizer`.  `.model` is the word segmentation model we need and `.vocab` is the vocabulary.

## Model

We use the open-source code [transformer-pytorch](http://nlp.seas.harvard.edu/2018/04/03/attention.html) developmented by Harvard.

## Requirements

This repo was tested on Python 3.6+ and PyTorch 1.5.1. The main requirements are:

- tqdm
- pytorch >= 1.5.1
- sacrebleu >= 1.4.14
- sentencepiece >= 0.1.94

To get the environment settled quickly, run:

```
pip install -r requirements.txt
```
## Usage

Hyperparameters can be modified in `config.py`.

To start training, please run:

```
python main.py
```

The training log is saved in `./experiment/train.log`, and the translation results of testing dataset is in `./experiment/output.txt`.
