import argparse
import sqlite3
from dta_ocr import create_progress_schema


def main():
    arg_parser = argparse.ArgumentParser(
        "match",
        description=(
            "Given a set of predictions generated with predict.py, " +
            "attempts to match the predicted lines to sections in " +
            "the original TEI file."
        )
    )
    arg_parser.add_argument(
        "--progress-file", dest="progress_file", default="progress.db",
        help=(
            "The location of a SQLite database which stores the progress " +
            "the dta_ocr scripts have made.\n" +
            "The database will be created if it does not exist!"
        )
    )
    args = arg_parser.parse_args()

    with sqlite3.connect(args.progress_file) as conn:
        create_progress_schema(conn)


if __name__ == "__main__":
    main()
