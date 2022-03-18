import argparse
from difflib import Match
import sqlite3
import fuzzysearch
import math
import functools
import time
import csv
from Levenshtein import distance
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from bs4 import BeautifulSoup
from dta_ocr import create_progress_schema
from dta_ocr.dta_tei_parser import DTADocument
from page.elements import PcGts, TextRegion


def schedule_matchings(conn: sqlite3.Connection):
    cursor = conn.execute(
        """SELECT dta_dirname, page_number FROM predictions
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


def fetch_scheduled_matchings(
    conn: sqlite3.Connection, only_finished: bool = True
) -> List[Matching]:
    cursor = conn.execute(
        f"""SELECT m.dta_dirname, m.page_number, p.prediction_path, d.tei_path
        FROM matchings m
        JOIN predictions p ON m.dta_dirname = p.dta_dirname
        JOIN documents d ON m.dta_dirname = d.dta_dirname
        {"WHERE m.status != 'finished'" if only_finished else ""}
        ORDER BY m.dta_dirname, m.page_number"""
    )
    return [
        Matching(dta_dirname, page_number, Path(pred_path), Path(tei_path))
        for dta_dirname, page_number, pred_path, tei_path in cursor
    ]


# given a single line from PRED and the corresponding page GT,
# returns the corresponding line in GT
def correct_line_with_gt(
    line: str, gt: str, max_norm_lev: float
) -> Optional[str]:
    if not line:
        return None

    max_lev: int = max(0, math.floor(max_norm_lev * len(gt)))
    print(f"Correcting line {line} with max_lev={max_lev}...")
    matches = fuzzysearch.find_near_matches(
        line, gt, max_l_dist=max_lev
    )
    if not matches:
        # not found in GT
        return None

    print(f"==> {matches[0].matched}")
    return matches[0].matched


@functools.lru_cache(maxsize=16)
def load_dta_doc(tei_path: Path) -> DTADocument:
    with tei_path.open("r") as file:
        soup = BeautifulSoup(file, "lxml")

    return DTADocument.from_tei_soup(soup)


# given a scheduled matching, compute the new GT lines
def match(
    matching: Matching, max_norm_lev: float
) -> Tuple[str, str]:
    if matching.pred_path.is_file():
        with matching.pred_path.open("r") as file:
            pcgts = PcGts.from_file(file)

        page = pcgts.page
        pred_lines: List[str] = [
            line.text.unicode
            for region in page.regions
            if isinstance(region, TextRegion)
            for line in region.lines
            if line.text is not None
        ]
    else:
        pred_lines = []

    doc = load_dta_doc(matching.tei_path)
    gt_text = doc.get_page_text(matching.page_number)
    print(pred_lines)

    pred_text = "\n".join(
        correct_line_with_gt(line, gt_text, max_norm_lev) or ""
        for line in pred_lines
    )

    return pred_text, gt_text


def perform_scheduled_matchings(
    db: sqlite3.Connection, scheduled: List[Matching]
):
    for matching in scheduled:
        pass


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
    arg_parser.add_argument(
        "--max-norm-lev", dest="max_norm_lev", default=0.0045
    )
    args = arg_parser.parse_args()

    with sqlite3.connect(args.progress_file) as conn:
        create_progress_schema(conn)
        schedule_matchings(conn)

        while True:
            scheduled = fetch_scheduled_matchings(conn)

            if scheduled:
                perform_scheduled_matchings(conn, scheduled)
            else:
                print("No matchings scheduled. Waiting for work...")
                time.sleep(10.0)

            schedule_matchings(conn)


if __name__ == "__main__":
    main()
