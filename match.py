import argparse
import sqlite3
import functools
import fuzzysearch
import time
import difflib
import multiprocessing
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Set
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
        f"""SELECT DISTINCT m.dta_dirname, p.page_number, p.prediction_path, d.tei_path
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


def correct_line_with_gt_ext(
    line: str, gt_segments: Set[str], cutoff: float
) -> Optional[str]:
    if not line:
        return None

    matches = []
    for gt_line in gt_segments:
        matches += fuzzysearch.find_near_matches(
            line, gt_line, max_l_dist=int(cutoff)
        )

    if matches:
        best_match: fuzzysearch.Match = min(matches, key=lambda m: m.dist)
        return best_match.matched
    else:
        return None


def correct_line_with_gt(
    line: Line, gt_lines: Set[str], cutoff: float
) -> Optional[Tuple[str, str]]:
    line_text = line.text.unicode
    if not line_text:
        return line.line_id, None

    matches = difflib.get_close_matches(
        line_text, gt_lines, n=1, cutoff=cutoff
    )
    if matches:
        gt_lines.remove(matches[0])
        print(f"{line_text} ===> {matches[0]}")
        return line.line_id, matches[0]
    else:
        return line.line_id, None


@functools.lru_cache(maxsize=16)
def load_dta_doc(tei_path: Path, intersperse: bool = False) -> DTADocument:
    with tei_path.open("r") as file:
        soup = BeautifulSoup(file, "lxml")

    return DTADocument.from_tei_soup(soup, intersperse)


# given a scheduled matching, compute the new GT lines
def match(
    matching: Matching, cutoff: float, intersperse: bool,
    pool: multiprocessing.Pool
) -> Dict[str, Optional[str]]:
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

    print("===============================")
    print(f"MATCHING {matching.dta_dirname} page {matching.page_number}")
    print("===============================")

    doc = load_dta_doc(matching.tei_path, intersperse=intersperse)
    gt_text = doc.get_page_text(matching.page_number)

    gt_lines: Set[str] = {
        gt_line.strip() for gt_line in gt_text.split("\n") if gt_line.strip()
    }

    return {
        line_id: correction
        for line_id, correction in pool.starmap(
            correct_line_with_gt, (
                (line, gt_lines, cutoff) for line in pred_lines
            )
        )
    }


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
        if not line_text:
            continue

        line = Line(
            line_id, pred_coords[line_id][0], pred_coords[line_id][1],
            Text(None, line_text, None)
        )
        page.regions.append(TextRegion(
            f"r{region_id}", pred_coords[line_id][0],
            [], None, [line]
        ))
        region_id += 1

    page.reading_order = None
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
    db: sqlite3.Connection, scheduled: List[Matching], cutoff: float,
    intersperse: bool, process_count: int
):
    with multiprocessing.Pool(process_count) as pool:
        for matching in scheduled:
            gt_lines = match(matching, cutoff, intersperse, pool)

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
    cpu_count = multiprocessing.cpu_count()

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
        "--cutoff", dest="cutoff", type=float, default=0.8,
        help=(
            "When matching, any substrings with a levenshtein distance " +
            "of this parameter or higher will be discarded. Higher values " +
            "allow more tolerance for mistakes in the prediction, but will " +
            "take significantly longer to process! Allowing more mistakes " +
            "also enables the possibility of matching unrelated strings."
        )
    )
    arg_parser.add_argument(
        "--intersperse", dest="intersperse", action="store_true",
        help=(
            "In some texts, e m p h a s i s is placed on words by " +
            "interspersing them with spaces. The prediction may or " +
            "may not pick up on this. When this flag is given, " +
            "the text in the TEI ground truth will be interspersed, " +
            "or kept without spaces otherwise."
        )
    )
    arg_parser.add_argument(
        "--process-count", dest="process_count", default=cpu_count, type=int,
        help=(
            "The amount of processes to use for computationally expensive " +
            "tasks. Recommended value is the system's CPU count " +
            f"(here: {cpu_count})."
        )
    )
    args = arg_parser.parse_args()

    with sqlite3.connect(args.progress_file) as conn:
        create_progress_schema(conn)
        schedule_matchings(conn)

        while True:
            scheduled = fetch_scheduled_matchings(conn, False)

            if scheduled:
                perform_scheduled_matchings(
                    conn, scheduled, args.cutoff,
                    args.intersperse, args.process_count
                )
            else:
                print("No matchings scheduled. Waiting for work...")
                time.sleep(10.0)

            schedule_matchings(conn)


if __name__ == "__main__":
    main()
