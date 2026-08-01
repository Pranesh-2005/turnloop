import argparse


def build_parser():
    parser = argparse.ArgumentParser(prog="app")
    parser.add_argument("--name", default="world")
    return parser


def main():
    args = build_parser().parse_args()
    print(f"hello {args.name}")


if __name__ == "__main__":
    main()
