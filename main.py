"""Diffuser CLI entry point. Dispatches to interactive or batch mode."""

import argparse

from batch import run_batch
from interactive import run_interactive


def main():
    parser = argparse.ArgumentParser(description="Diffuser CLI — generate images from text using diffusion models")
    parser.add_argument(
        "--batch",
        metavar="BATCH_JSON",
        help="Run in batch mode using a JSON file of prompts (skips interactive mode)",
    )
    args = parser.parse_args()

    if args.batch:
        run_batch(args.batch)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
