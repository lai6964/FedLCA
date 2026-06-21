import argparse
import datetime
import os
import sys
import traceback

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

    parser.add_argument("--dataset", type=str, default="officehome",
                        choices=[
                            "cifar10",
                            "cifar100",
                            "tinyimagenet",
                            "domainnet",
                            "officehome",
                        ])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--download_data", action="store_true")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["cnn", "fedavgcnn", "resnet18"])


    parser.add_argument("--init_with_fedavg", type=str_to_bool, default=False)
    parser.add_argument("--mode", type=str, default="traditional",
                        choices=["traditional", "tranditional", "personal"])
    parser.add_argument("--init_checkpoint", type=str, default="")
    parser.add_argument("--save_checkpoint", type=str, default="")

    parser.add_argument("--method", type=str, default="ours",
                        choices=[
                            "fedavg",
                            "topk_params",
                            "sequential_conv",
                            "server_top_importance_conv",
                            "server_only",
                            "ours",
                            "only_importance",
                            "only_consistency",
                            "wo_importance",
                            "wo_consistency",
                            "wo_staleness",
                            "wo_client_adaptive",
                        ])

    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--client_frac", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--train_ratio", type=float, default=0.75)
    parser.add_argument("--partition_dir", type=str, default="./data/partitions")
    parser.add_argument("--regenerate_partition", type=str_to_bool, default=False)

    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=256)

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--server_layer_budget", type=int, default=4)
    parser.add_argument("--client_layer_budget", type=int, default=2)
    parser.add_argument("--topk_param_ratio", type=float, default=0.2)
    parser.add_argument("--staleness_threshold", type=int, default=5)

    parser.add_argument("--min_clients_per_layer", type=int, default=5)
    parser.add_argument("--small_layer_lr", type=float, default=0.5)

    parser.add_argument("--stat_beta", type=float, default=0.9)
    parser.add_argument("--resource_mode", type=str, default="moderate",
                        choices=["mild", "moderate", "severe"])

    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    if args.mode == "tranditional":
        args.mode = "traditional"
    return args


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def get_run_log_path(args):
    log_name = f"{args.method}_{args.dataset}_a{args.alpha}.log"
    return os.path.join(args.output_dir, log_name)


def print_run_settings(args):
    print("=" * 80)
    print("Run settings")
    print("=" * 80)
    print(f"Started at: {datetime.datetime.now().isoformat(timespec='seconds')}")
    print(f"Command: {' '.join(sys.argv)}")
    print(f"Working directory: {os.getcwd()}")
    print()
    for key, value in sorted(vars(args).items()):
        print(f"{key}: {value}")
    print("=" * 80)
    print()


def run_with_logging(args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = get_run_log_path(args)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            print(f"Log file: {log_path}")
            print_run_settings(args)
            print(args.method)
            run_experiment(args)
            print()
            print(f"Finished at: {datetime.datetime.now().isoformat(timespec='seconds')}")
            print(f"Log saved to: {log_path}")
        except Exception:
            print()
            print("Run failed with exception:")
            traceback.print_exc()
            raise
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    args = parse_args()
    run_with_logging(args)
