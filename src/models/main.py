from .train import train_probes


def main():
    train_loss, train_auc, val_loss, val_auc, model = train_probes()

    


if __name__ == "__main__":
    main()