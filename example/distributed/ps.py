import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', type=str)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.role == 'ps':
        pass
    else:
        pass


if __name__ == '__main__':
    main()
