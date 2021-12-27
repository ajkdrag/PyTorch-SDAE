from train import Trainer
from argparse import ArgumentParser
from utils.config_parser import ConfigParser

def parse_flags():
    parser = ArgumentParser()
    parser.add_argument("--hyps", default=str, help="Path to hyperparams file")
    parser.add_argument("--opts", default=str, help="Path to options file")
    return parser.parse_args()


def main(FLAGS):
    print(FLAGS)
    hyps, opts = ConfigParser.parse_configs(FLAGS.hyps, FLAGS.opts)
    
    trainer = Trainer(opts, hyps)
    trainer.setup()
    trainer.run()


if __name__ == "__main__":
    opt = parse_flags()
    main(opt)
