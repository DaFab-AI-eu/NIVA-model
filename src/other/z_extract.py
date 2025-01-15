import argparse
import py7zr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)

    args = parser.parse_args()

    # multi volume
    # with multivolumefile.open(args.input, mode='rb') as target_archive:
    #     with SevenZipFile(path=args.output, 'r') as archive:
    #         archive.extractall() # extracts into current directory
    # python -m py7zr x monty.7z target-dir/ # doen't work with multi-volume
    with py7zr.SevenZipFile(args.input, mode='r') as z:
        z.extractall(path=args.output)