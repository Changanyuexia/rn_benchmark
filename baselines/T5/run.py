#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DRG_parsing 
@File ：run.py
@Author ：xiao zhang
@Date ：2022/11/14 12:27
'''

import argparse
import os
import sys
sys.path.append(".")

from model import get_dataloader, Generator

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lang", required=False, type=str, default="en",
                        help="language in [en, nl, de ,it]")
    parser.add_argument("-t", "--train", required=False, type=str,
                        default=os.path.join(path, "data/rn/test.csv"),
                        help="train sets")
    parser.add_argument("-d", "--dev", required=False, type=str,
                        default=os.path.join(path, "data/rn/val.csv"),
                        help="dev sets")
    parser.add_argument("-e", "--test", required=False, type=str,
                        default=os.path.join(path, "data/rn/test.csv"),
                        help="standard test sets")
    parser.add_argument("-mp", "--model_path", required=False, type=str,
                        default="",
                        help="path to load the trained model")
    parser.add_argument("-s", "--save", required=False, type=str,
                        default=os.path.join(path, "result/T5/rn"),
                        help="path to save the result")
    parser.add_argument("-epoch", "--epoch", required=False, type=int,
                        default=16)
    parser.add_argument("-lr", "--learning_rate", required=False, type=float,
                        default=1e-04)
    parser.add_argument("-ms", "--model_save", required=False, type=str,
                        default=os.path.join(path, "model/T5/rn"))
    args = parser.parse_args()
    return args


def ensure_directory(path):
    """ Ensure directory exists. """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def main():
    args = create_arg_parser()

    # Ensure save directories exist
    ensure_directory(args.save)
    ensure_directory(args.model_save)

    # Train process
    lang = args.lang

    # Train loader
    train_dataloader = get_dataloader(args.train)

    # Test loader
    test_dataloader = get_dataloader(args.test)
    dev_dataloader = get_dataloader(args.dev)

    # Hyperparameters
    epoch = args.epoch
    lr = args.learning_rate

    # load the model
    if os.path.exists(args.model_path):
        model_path = args.model_path
    else:
        model_path = ""

    # Train
    bart_classifier = Generator(lang, load_path=model_path)
    bart_classifier.train(train_dataloader, dev_dataloader, lr=lr, epoch_number=epoch, save_path=args.model_save)

    # Standard test
    trained_bart_classifier = Generator(lang, load_path=args.model_save)
    trained_bart_classifier.evaluate(test_dataloader, os.path.join(args.save, "predictions.csv"))


if __name__ == '__main__':
    main()
