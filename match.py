import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List
from dta_ocr import create_progress_schema
from tei_reader import TeiReader
from page.elements import PcGts, TextRegion


def schedule_matchings(conn: sqlite3.Connection):
    cursor = conn.execute(
        """SELECT dta_dirname, page_number
        FROM predictions
        WHERE status = 'finished'"""
    )

    conn.executemany(
        """INSERT OR IGNORE INTO matchings
            ( dta_dirname, page_number, gt_path, status )
        VALUES
            ( ?, ?, NULL, 'pending' )""",
        cursor
    )
    conn.commit()


@dataclass
class Matching:
    dta_dirname: str
    page_number: int
    pred_path: Path
    tei_path: Path


def fetch_scheduled_matchings(conn: sqlite3.Connection) -> List[Matching]:
    cursor = conn.execute(
        """SELECT m.dta_dirname, m.page_number, p.prediction_path, d.tei_path
        FROM matchings m
        NATURAL JOIN predictions p
        JOIN documents d ON m.dta_dirname = d.dta_dirname
        WHERE m.status != 'finished'
        ORDER BY m.dta_dirname, m.page_number"""
    )
    return [
        Matching(dta_dirname, page_number, Path(pred_path), Path(tei_path))
        for dta_dirname, page_number, pred_path, tei_path in cursor
    ]


def match(matching: Matching):
    with matching.pred_path.open("r") as file:
        pcgts = PcGts.from_file(file)

    page = pcgts.page
    pred_lines: List[str] = [
        line.unicode
        for region in page.regions
        if isinstance(region, TextRegion)
        for line in region.lines
        if line.text is not None
    ]

    tei_reader = TeiReader()
    tei = tei_reader.read_file(str(matching.tei_path))
    gt_text: str = tei.text


def main():
    arg_parser = argparse.ArgumentParser(
        "match",
        description=(
            "Given a set of predictions generated with predict.py, " +
            "attempts to match the predicted lines to sections in " +
            "the original TEI file, and outputs a ground truth."
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
