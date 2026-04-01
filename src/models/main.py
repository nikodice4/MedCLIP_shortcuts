# main.py: script that calls the train_probes

import argparse
from .train import train_probes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="chestxray", choices=["chestxray", "padchest"])
    args = parser.parse_args()

    train_loss, train_auc, val_loss, val_auc, model = train_probes(dataset=args.dataset)

if __name__ == "__main__":
    main()