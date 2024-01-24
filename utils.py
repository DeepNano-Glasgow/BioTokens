import os
import logging
import sentencepiece as spm


def amino_tokenizer_load():
    sp_ami = spm.SentencePieceProcessor()
    sp_ami.Load('{}.model'.format("./tokenizer/ami"))
    return sp_ami


def measurements_tokenizer_load():
    sp_meas = spm.SentencePieceProcessor()
    sp_meas.Load('{}.model'.format("./tokenizer/meas"))
    return sp_meas


def set_logger(log_path):
   
    if os.path.exists(log_path) is True:
        os.remove(log_path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(meassage)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(meassage)s'))
        logger.addHandler(stream_handler)


