import argparse
import sqlite3
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import List
from bs4 import BeautifulSoup
from fuzzysearch import find_near_matches
# from dta_ocr import create_progress_schema
from dta_ocr.dta_tei_parser import DTADocument
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


# given a single line from PRED and the entirety of GT,
# returns the corresponding line in GT
def correct_line_with_gt(line: str, gt: str) -> str:
    matcher = difflib.SequenceMatcher(None, line, gt)

    for match in matcher.get_matching_blocks():
        print(gt[match.b:match.b+match.size])

    # TODO: Implement.
    return ""


# given a scheduled matching, compute the new GT lines
def match(matching: Matching) -> List[str]:
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

    gt_text = ""

    return [
        correct_line_with_gt(line, gt_text) for line in pred_lines
    ]


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
    # args = arg_parser.parse_args()

    with open(
        "dta_komplett_2021-05-13/davidis_kochbuch_1879.TEI-P5.xml", "r"
    ) as file:
        soup = BeautifulSoup(file, "lxml")

    dta_doc = DTADocument.from_tei_soup(soup)
    print(dta_doc.get_page_text(8))

    matches = find_near_matches(
        "Ueberfluſſes, des Verſchwendens iſt vorüber, ſ o l l t e wenigſtens",
        dta_doc.get_page_text(8),
        max_l_dist=15
    )

    print(matches)

    # with sqlite3.connect(args.progress_file) as conn:
    #    create_progress_schema(conn)


if __name__ == "__main__":
    main()
