import json
import logging
import os

import torch
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.training.util import evaluate

from prepare import (prepare_config, prepare_model, prepare_optimizer,
                     prepare_positive_dastaset, prepare_text_field_emebedder,
                     prepare_token_indexers)

logger = logging.getLogger(__name__)


def main():
    config = prepare_config()
    model_path =  os.path.join(config['DEFAULT']['model_path'], "_".join([
        config['DEFAULT']['model_name'], config['DEFAULT']['type'], config['DEFAULT']['alpha'], 
        config['DEFAULT']['beta'], config['DEFAULT']['T'], config['DEFAULT']['cuda_device']
    ]))
    model_path = config['DEFAULT']['model_path']

    token_indexers = prepare_token_indexers(config)
    text_field_embedder = prepare_text_field_emebedder(config)
    positive_dataset, vocab, positive_dataset_reader = prepare_positive_dastaset(config, token_indexers)

    logger.info(str(vocab))
    
    p = positive_dataset_reader.M / positive_dataset_reader.N

    model = prepare_model(config, vocab, text_field_embedder, positive_dataset_reader, model_path)
    device = int(config['DEFAULT']['cuda_device'])
    iterator = BucketIterator(sorting_keys=[('text', 'num_tokens')], batch_size=int(config['DEFAULT']['batch_size']))
    iterator.index_with(vocab)
    with open(os.path.join(model_path, "config.ini"), "w", encoding="utf-8") as fout:
        config.write(fout)
    
    optimizer = prepare_optimizer(config, model)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=positive_dataset,
        validation_dataset=positive_dataset,
        patience=3,
        validation_metric='-loss',
        num_serialized_models_to_keep=0,
        num_epochs=int(config['DEFAULT']['num_epochs']),
        serialization_dir=model_path,
        cuda_device=device
    )
    best_epoch = None
    if config['DEFAULT']['do_train'] == 'yes':
        model.to(device)
        ret = trainer.train()
        best_epoch = ret['best_epoch']
    
    if config['DEFAULT']['do_eval'] == 'yes':
        model.load_state_dict(torch.load(os.path.join(model_path, "best.th")))
        model.to(device)
        result = evaluate(model=model,
                        instances=positive_dataset,
                        data_iterator=iterator,
                        cuda_device=device,
                        batch_weight_key="")
        result['best_epoch'] = best_epoch

        with open(os.path.join(model_path, "result.json"), "w", encoding="utf-8") as fout:
            json.dump(result, fout)


if __name__ == "__main__":
    main()
