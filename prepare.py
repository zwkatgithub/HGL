import argparse
import configparser

import torch.optim as optim
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedBertEmbedder

from hyper_denosing.dataset_readers import PositiveDatasetReader
from hyper_denosing.models import HyperModel


def prepare_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", dest="config_file")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config_file)

    return config

def prepare_token_indexers(config):
    token_indexers = {
        "bert": PretrainedBertIndexer(config['DEFAULT']['pretrained_model'],
                use_starting_offsets=config['token_indexers']['use_starting_offsets']=='yes',
                do_lowercase=config['token_indexers']['do_lowercase']=='yes')
    }
    return token_indexers

def prepare_text_field_emebedder(config):
    token_embedders = {
        "bert": PretrainedBertEmbedder(config['DEFAULT']['pretrained_model'],
                requires_grad=False,
                top_layer_only=True)
    }
    embedder_to_indexer_map = {"bert":["bert", "bert-offsets"]}
    text_field_embedder = BasicTextFieldEmbedder(token_embedders,
                            embedder_to_indexer_map, allow_unmatched_keys=True)
    return text_field_embedder

def prepare_model(config, vocab, text_field_embedder, positive_dataset_reader, model_path):
    model_class = {"hyper": HyperModel}[config['DEFAULT']['model_name']]

    model = model_class(vocab,
                        text_field_embedder,
                        int(config['DEFAULT']['hidden_size']),
                        positive_dataset_reader.N,
                        positive_dataset_reader.M,
                        positive_dataset_reader.golden_N,
                        int(config['DEFAULT']['batch_size']),
                        model_path,
                        float(config['DEFAULT']['T']),
                        )
    return model

def prepare_positive_dastaset(config, token_indexers):
    positive_dataset_reader = PositiveDatasetReader(token_indexers,
                                 config['DEFAULT']['type'])
    positive_dataset = positive_dataset_reader.read(config['DEFAULT']['train_dataset'])
    vocab = Vocabulary.from_instances(positive_dataset)
    return positive_dataset, vocab, positive_dataset_reader

def prepare_optimizer(config, model):
    optimizer = optim.Adam(filter(lambda x:x.requires_grad, model.parameters()), lr=float(config['DEFAULT']['lr']))
    return optimizer
