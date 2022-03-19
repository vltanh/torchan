import pprint
import argparse

import yaml

from core.utils.random_seed import set_determinism
from core.utils.getter import get_data
from core.trainers import SupervisedTrainer


def train(config):
    assert config is not None, "Do not have config file!"

    pprint.PrettyPrinter(indent=2).pprint(config)

    # 1: Load datasets
    train_dataloader, val_dataloader = \
        get_data(config['dataset'], config['seed'])

    # 2: Create trainer
    trainer = SupervisedTrainer(config=config)

    # 3: Start trainining
    trainer.train(train_dataloader=train_dataloader,
                  val_dataloader=val_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config['gpus'] = args.gpus
    config['debug'] = args.debug

    set_determinism()
    train(config)
