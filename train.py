import argparse

from experiment import run_experiment


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value: true or false")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cifar100",
                        choices=["cifar10", "cifar100"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--download_data", action="store_true")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["cnn", "resnet18"])
    parser.add_argument("--init_with_fedavg", type=str_to_bool, default=False)
    parser.add_argument("--train_fc_first", type=str_to_bool, default=False)
    parser.add_argument("--init_checkpoint", type=str, default="")
    parser.add_argument("--save_checkpoint", type=str, default="")

    parser.add_argument("--method", type=str, default="ours",
                        choices=[
                            "fedavg",
                            "server_only",
                            "ours",
                            "wo_importance",
                            "wo_consistency",
                            "wo_staleness",
                            "wo_client_adaptive",
                        ])

    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--client_frac", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.1)

    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=256)

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--server_layer_budget", type=int, default=4)
    parser.add_argument("--client_layer_budget", type=int, default=2)
    parser.add_argument("--staleness_threshold", type=int, default=5)

    parser.add_argument("--min_clients_per_layer", type=int, default=2)
    parser.add_argument("--small_layer_lr", type=float, default=0.5)

    parser.add_argument("--stat_beta", type=float, default=0.9)
    parser.add_argument("--resource_mode", type=str, default="moderate",
                        choices=["mild", "moderate", "severe"])

    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args.method)
    run_experiment(args)
