import argparse
import sqlite3
import fuzzysearch
import math
import functools
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from bs4 import BeautifulSoup
from dta_ocr import create_progress_schema
from dta_ocr.dta_tei_parser import DTADocument
from page.elements import PcGts, TextRegion, Line, Coordinates, Text, Baseline


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
    matches = fuzzysearch.find_near_matches(
        line, gt, max_l_dist=max_lev
    )
    if not matches:
        # not found in GT
        return None

    return matches[0].matched


@functools.lru_cache(maxsize=16)
def load_dta_doc(tei_path: Path) -> DTADocument:
    with tei_path.open("r") as file:
        soup = BeautifulSoup(file, "lxml")

    return DTADocument.from_tei_soup(soup)


# given a scheduled matching, compute the new GT lines
def match(
    matching: Matching, max_norm_lev: float
) -> Dict[str, str]:
    if matching.pred_path.is_file():
        with matching.pred_path.open("r") as file:
            pcgts = PcGts.from_file(file)

        page = pcgts.page
        pred_lines: List[Line] = [
            line
            for region in page.regions
            if isinstance(region, TextRegion)
            for line in region.lines
            if line.text is not None
        ]
    else:
        pred_lines: List[Line] = []

    doc = load_dta_doc(matching.tei_path)
    gt_text = doc.get_page_text(matching.page_number)

    gt_lines = {
        line.line_id: correct_line_with_gt(
            line.text.unicode, gt_text, max_norm_lev
        )
        for line in pred_lines
    }

    return gt_lines


def write_lines_to_pagexml(
    lines: Dict[str, str], original: PcGts, path: Path
):
    original.metadata.last_change = datetime.now()
    page = original.page

    pred_coords: Dict[str, Tuple[Coordinates, Optional[Baseline]]] = {
        line.line_id: (line.coords, line.baseline)
        for region in page.regions
        if isinstance(region, TextRegion)
        for line in region.lines
    }
    page.regions = []

    region_id = 0
    for line_id, line_text in lines.items():
        line = Line(
            line_id, pred_coords[line_id][0], pred_coords[line_id][1],
            Text(None, line_text, None)
        )
        page.regions.append(TextRegion(
            f"r{region_id}", pred_coords[line_id][0],
            [], None, [line]
        ))
        region_id += 1

    original.save_to_file(path)


# Path("abc.pred.xml").stem will yield "abc.pred".
# innermost_stem produces "abc" instead
def innermost_stem(path: Path) -> Path:
    path_stem = path.stem

    while path.suffixes:
        path = Path(path_stem)
        path_stem = path.stem

    return path_stem


def perform_scheduled_matchings(
    db: sqlite3.Connection, scheduled: List[Matching], max_norm_lev: float
):
    for matching in scheduled:
        gt_lines = match(matching, max_norm_lev)

        if not matching.pred_path.is_file():
            print(f"Warning: Path {matching.pred_path} is not a file.")
            continue

        with matching.pred_path.open("r") as file:
            pcgts = PcGts.from_file(file)

        write_lines_to_pagexml(
            gt_lines, pcgts,
            matching.pred_path.parent /
            f"{innermost_stem(matching.pred_path)}.gt.xml"
        )


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
        "--max-norm-lev", dest="max_norm_lev", type=float, default=0.0045
    )
    args = arg_parser.parse_args()

    with sqlite3.connect(args.progress_file) as conn:
        create_progress_schema(conn)
        schedule_matchings(conn)

        while True:
            scheduled = fetch_scheduled_matchings(conn)

            if scheduled:
                perform_scheduled_matchings(conn, scheduled, args.max_norm_lev)
            else:
                print("No matchings scheduled. Waiting for work...")
                time.sleep(10.0)

            schedule_matchings(conn)


if __name__ == "__main__":
    main()
