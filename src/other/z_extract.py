import argparse
import py7zr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)

    args = parser.parse_args()

    # multi volume
    # with multivolumefile.open('example.7z', mode='rb') as target_archive:
    #     with SevenZipFile(target_archive, 'r') as archive:
    #         archive.extractall()
    # python -m py7zr x monty.7z target-dir/
    with py7zr.SevenZipFile(args.input, mode='r') as z:
        z.extractall(path=args.output)