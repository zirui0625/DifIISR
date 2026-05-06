import argparse
from omegaconf import OmegaConf
from trainer import TrainerDifIISR as Trainer

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./logs/DifIISR",
        help="Folder to save the checkpoints and training log.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="Path to a checkpoint to resume from. If empty, train from scratch.",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="./configs/DifIISR_train.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=15,
        help="Number of diffusion steps (overrides diffusion.params.steps).",
    )
    return parser.parse_args()


def main():
    args = get_parser()

    configs = OmegaConf.load(args.cfg_path)
    configs.diffusion.params.steps = args.steps

    for key in ("cfg_path", "save_dir", "resume", "steps"):
        configs[key] = getattr(args, key)

    trainer = Trainer(configs)
    trainer.train()


if __name__ == "__main__":
    main()