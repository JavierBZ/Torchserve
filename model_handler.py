import torch
import torch.nn as nn

import pandas as pd

import os

import torch

from pre_proccessing_tweets import clean

from transformers import BertModel, BertTokenizer, AdamW

from ts.torch_handler.base_handler import BaseHandler

"""
ModelHandler defines a custom model handler.
"""


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self.model = None
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0

    def initialize(self, ctx):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.properties = ctx.system_properties
        self.initialized = True
        #  load the model, refer 'c     ustom handler class' above for details
        self.device = torch.device("cuda:" + str(self.properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        model_dir = self.properties.get("model_dir")

        # Read model serialize/pt file
        model_pt_path = os.path.join(model_dir, "model.bin")
        # # Read model definition file
        # model_def_path = os.path.join(model_dir, "model.py")
        # if not os.path.isfile(model_def_path):
        #     raise RuntimeError("Missing the model definition file")
        PRE_TRAINED_MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-cased'

        from model import SentimentClassifier

        self.model = SentimentClassifier(3)
        self.model.to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

        self.model.load_state_dict(torch.load(model_pt_path,map_location=torch.device(self.device)))

        self.initialized = True

        print('CARGADOOOO')
    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")

        tweet = text.decode('utf-8')
        preprocessed_tweet = clean(tweet)

        MAX_LEN = 50
        inputs = self.tokenizer.encode_plus(
                  preprocessed_tweet,
                  add_special_tokens=True,
                  max_length= MAX_LEN,
                  return_token_type_ids=False,
                  padding='max_length',
                  truncation  = True,
                  return_attention_mask=True,
                  return_tensors='pt',
                )

        return {
                  'review_text': tweet,
                  'input_ids': inputs['input_ids'].flatten(),
                  'attention_mask': inputs['attention_mask'].flatten()
                }


    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output

        input_ids = model_input["input_ids"].to(self.device)
        attention_mask = model_input["attention_mask"].to(self.device)


        model_output = self.model(
          input_ids=input_ids.unsqueeze(0),
          attention_mask=attention_mask.unsqueeze(0)
        )

        _, preds = torch.max(model_output, dim=1)


        predicted_idx = str(preds.item())
        return [predicted_idx]

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        postprocess_output = inference_output
        return postprocess_output

_service = ModelHandler()

def handle(data, context):

    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e
