from argparse import ArgumentParser

def parse_opt():
    parser = ArgumentParser()
    opt = parser.parse_args()
    return opt

def main(opt):
    print(opt)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
