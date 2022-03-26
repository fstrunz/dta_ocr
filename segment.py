import argparse
import multiprocessing
import subprocess
import os
import itertools
import torch
import sqlite3
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, TypeVar, Optional
from PIL import Image
from kraken import blla
from page.elements import PcGts, Page, Metadata, Region, Point
from page.elements import Line, TextRegion, RegionRefIndexed
from page.elements.coords import Coordinates, Baseline
from page.elements.reading_order import ReadingOrder
from page.elements.reading_order.ordered_group import OrderedGroup
from segmentation.postprocessing.layout_settings import LayoutProcessingMethod
from segmentation.preprocessing.source_image import SourceImage
from segmentation.predictors import PredictionSettings, Predictor
from segmentation.scripts.layout import (
    process_layout, LayoutProcessingSettings
)
from dta_ocr import create_progress_schema

T = TypeVar("T")


def filename_without_exts(path: Path) -> str:
    filename = path.name
    dot_index = filename.index(".")
    return filename[:dot_index]


def chunked(iterable: Iterable[T], n: int) -> Iterable[List[T]]:
    it = iter(iterable)

    while True:
        # take next n elements
        chunk = list(itertools.islice(it, n))

        # we are done
        if not chunk:
            return

        yield chunk


def facsimiles(facsimile_path: Path) -> Iterable[Tuple[Path, Path]]:
    for doc_path in facsimile_path.iterdir():
        if doc_path.is_dir():
            for fac_path in doc_path.iterdir():
                if (
                    fac_path.is_file() and
                    not fac_path.name.endswith(".bin.png") and
                    not fac_path.name.endswith(".nrm.png") and
                    not fac_path.name.endswith(".xml")
                ):
                    yield fac_path, doc_path


def schedule_downloaded_facsimiles(conn: sqlite3.Connection):
    print("Scheduling downloaded facsimiles for segmentation...")
    cursor = conn.execute(
        """SELECT dta_dirname, page_number FROM facsimiles
        WHERE status = 'finished'"""
    )

    conn.executemany(
        """INSERT OR IGNORE INTO segmentations
            ( dta_dirname, page_number, segmenter,
            model_path, file_path, status )
        VALUES
            ( ?, ?, NULL, NULL, NULL, 'pending' )""",
        cursor
    )

    conn.commit()


@dataclass
class Facsimile:
    dta_dirname: str
    page_number: int

    def path(self, facsimile_path: Path) -> Path:
        return (
            facsimile_path / self.dta_dirname / f"{str(self.page_number)}.jpg"
        )


def scheduled_facsimiles(
    conn: sqlite3.Connection
) -> Iterable[Facsimile]:
    print("Fetching scheduled facsimiles...")
    cursor = conn.execute(
        """SELECT dta_dirname, page_number FROM segmentations
        WHERE status != 'finished'
        ORDER BY dta_dirname, page_number"""
    )

    return (Facsimile(d, p) for d, p in cursor)


def save_segmented_facsimile(
    conn: sqlite3.Connection, facsimile: Facsimile,
    segmenter: str, model_path: Optional[Path], seg_path: Path
):
    conn.execute(
        """UPDATE segmentations
        SET status = 'finished',
            segmenter = ?,
            model_path = ?,
            file_path = ?
        WHERE dta_dirname = ? AND page_number = ?;
        """,
        (
            segmenter, None if model_path is None else str(model_path),
            str(seg_path), facsimile.dta_dirname, facsimile.page_number
        )
    )
    conn.commit()


def binarise_facsimile_chunk(
    ocropy_venv: Path, chunk_paths: List[Path]
) -> bool:
    python_path = (ocropy_venv / "bin" / "python").resolve()
    nlbin_path = (ocropy_venv / "bin" / "ocropus-nlbin").resolve()
    arguments = [python_path, nlbin_path]
    arguments += chunk_paths

    with subprocess.Popen(arguments) as nlbin_proc:
        exit_code = nlbin_proc.wait()
        if exit_code:
            print(
                f"ocropy-nlbin failed with exit code {exit_code}, " +
                f"given arguments {arguments}"
            )
            return False

    for fac_path in chunk_paths:
        bin_dir = fac_path.parent / "bin"
        bin_dir.mkdir(exist_ok=True)

        name = filename_without_exts(fac_path)
        bin_path = fac_path.parent / f"{name}.bin.png"
        nrm_path = fac_path.parent / f"{name}.nrm.png"

        if bin_path.is_file():
            bin_path.rename(bin_dir / bin_path.name)
        else:
            print(f"Bin file \"{bin_path}\" not found!!")

        if nrm_path.is_file():
            nrm_path.rename(bin_dir / nrm_path.name)
        else:
            print(f"Nrm file \"{nrm_path}\" not found!!")

    return True


def binarise_facsimiles(
    ocropy_venv: Path, facsimile_path: Path, process_count: int
):
    facsimile_count = sum(1 for _ in facsimiles(facsimile_path))

    chunk_size = facsimile_count // process_count
    print(f"Processing {facsimile_count} in {chunk_size} chunks " +
          f"over {process_count} processes...")

    with multiprocessing.Pool(process_count) as pool:
        pool.starmap(
            binarise_facsimile_chunk,
            ((ocropy_venv, chunk) for chunk in chunked(
                (fac for fac, _ in facsimiles(facsimile_path)),
                facsimile_count // process_count
            ))
        )


# computes regions in reading order
def segmentation_to_regions(seg: Dict[str, Any]) -> List[Region]:
    line_id = 0
    regions = []

    for line in seg["lines"]:
        baseline = Baseline([Point(p[0], p[1]) for p in line["baseline"]])
        boundary = Coordinates([Point(p[0], p[1]) for p in line["boundary"]])

        text_line = Line(f"l{line_id}", boundary, baseline, None)
        region = TextRegion(f"r{line_id}", boundary, [], None, [text_line])
        regions.append(region)

        line_id += 1

    return regions


def create_reading_order(regions: List[Region]) -> ReadingOrder:
    region_refs: List[str] = [
        RegionRefIndexed(r.region_id, i) for i, r in enumerate(regions)
    ]
    ordered_group = OrderedGroup("ro1", region_refs, "Region reading order")
    return ReadingOrder(ordered_group)


def find_binarisation_path(fac_path: Path, parent_dir: Path) -> Optional[Path]:
    fac_name = fac_path.stem
    bin_path = parent_dir / "bin" / f"{fac_name}.bin.png"
    if not bin_path.is_file():
        return None
    return bin_path


def segment_facsimile_kraken(
    conn: sqlite3.Connection, fac: Facsimile, fac_path: Path
):
    parent_dir = fac_path.parent
    fac_name = fac_path.stem
    bin_path = find_binarisation_path(fac_path, parent_dir)

    if bin_path is None:
        print(f"Could not find binarisation for {fac_path}!!")
        return

    print(f"Segmenting {bin_path}...")
    with Image.open(bin_path) as bin_img:
        bin_width, bin_height = bin_img.width, bin_img.height
        segmentation = blla.segment(bin_img)

    now = datetime.now()
    metadata = Metadata(
        "segment.py", now, now,
        "Generated using Kraken baseline segmentation."
    )

    regions = segmentation_to_regions(segmentation)
    reading_order = create_reading_order(regions)

    page = Page(
        (bin_width, bin_height), bin_path.name,
        reading_order, regions
    )
    pcgts = PcGts(None, metadata, page)

    seg_file = parent_dir / "bin" / f"{fac_name}.xml"
    pcgts.save_to_file(seg_file)
    save_segmented_facsimile(
        conn, fac, "kraken", None, seg_file
    )


def segment_facsimiles_kraken(
    conn: sqlite3.Connection, facsimile_path: Path,
    facsimiles: List[Facsimile]
):
    for fac in facsimiles:
        segment_facsimile_kraken(conn, fac, fac.path(facsimile_path))


def segment_facsimile_i6(
    conn: sqlite3.Connection, fac_path: Path, predictor: Predictor,
    pool: multiprocessing.Pool, fac: Facsimile, model_path: Path
):
    print(f"Segmenting {fac_path}...")

    img = SourceImage.load(fac_path)
    img.pil_image = img.pil_image.convert("RGB")

    prediction, scaled_image = predictor.predict_image(img, process_pool=pool)
    layout_settings = LayoutProcessingSettings(
        marginalia_postprocessing=False,
        source_scale=True,
        layout_method=LayoutProcessingMethod.FULL
    )

    analysed_content = process_layout(
        prediction, scaled_image, pool, layout_settings
    )
    analysed_content = analysed_content.to_pagexml_space(
        prediction.prediction_scale_factor
    )
    xml_gen = analysed_content.export(
        scaled_image, fac_path, simplified_xml=False
    )

    xml_gen.save_textregions_as_xml(str(fac_path.parent))
    xml_path = fac_path.parent / f"{fac_path.stem}.xml"
    save_segmented_facsimile(
        conn, fac, "segmentation-pytorch", model_path, xml_path
    )


def segment_facsimiles_i6(
    conn: sqlite3.Connection, facsimile_path: Path, model_path: Path,
    process_count: int, facsimiles: List[Facsimile]
):
    print("Setting up predictor...")
    settings = PredictionSettings([model_path], 1000000, None, None)
    predictor = Predictor(settings)

    if not torch.cuda.is_available():
        torch.set_num_threads(process_count)

    with multiprocessing.Pool(process_count) as pool:
        for fac in facsimiles:
            segment_facsimile_i6(
                conn, fac.path(facsimile_path), predictor, pool,
                fac, model_path
            )


def segment_facsimiles(
    conn: sqlite3.Connection,
    facsimile_path: Path, process_count: int, ocropy_venv: Optional[Path],
    segmenter: str, model_path: Path, facsimiles: List[Facsimile]
):
    print("Segmenting facsimiles...")

    if segmenter == "kraken":
        print("Binarising facsimiles with OCRopus...")
        binarise_facsimiles(
            ocropy_venv, facsimile_path, process_count
        )
        segment_facsimiles_kraken(conn, facsimile_path, facsimiles)
    elif segmenter == "i6":
        # This segmenter automatically does binarisation
        segment_facsimiles_i6(
            conn,
            facsimile_path, model_path, process_count, facsimiles
        )


def main():
    cpu_count = multiprocessing.cpu_count()

    arg_parser = argparse.ArgumentParser(
        "segment",
        description=(
            "Given a set of facsimiles, in the same format as the output " +
            "of download.py, preprocesses (binarisation, " +
            "deskewing) and segments all images."
        )
    )
    arg_parser.add_argument(
        "--ocropy-venv", dest="ocropy_venv", default=None,
        help=(
            "Path to a virtualenv containing a Python 2.7 install along with " +
            "the OCRopy binaries. Required when '--segmenter kraken' is given."
        )
    )
    arg_parser.add_argument(
        "--segmenter", dest="segmenter", default="i6", type=str,
        choices=["kraken", "i6"],
        help=(
            "The segmenter to use for segmentation."
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
    arg_parser.add_argument(
        "--facsimile-dir", dest="facsimile_dir", default="facsimiles",
        help=(
            "A directory containing document subdirectories, which contain " +
            "page numbered facsimiles, analogous to the output of " +
            "download.py."
        )
    )
    arg_parser.add_argument(
        "--model-path", dest="model_path", default=None,
        help=(
            "The model path for segmentation-pytorch. Required when " +
            "'--segmenter i6' is specified."
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

    if args.ocropy_venv is None:
        if args.segmenter == "kraken":
            print(
                "An OCRopy venv directory must be specified " +
                "when using Kraken!!"
            )
            exit(1)

        ocropy_venv = None
    else:
        ocropy_venv = Path(args.ocropy_venv)
        if not ocropy_venv.is_dir():
            print("Given OCRopy venv directory is not a directory!")
            exit(1)

        python_path = ocropy_venv / "bin" / "python"
        if not python_path.is_file():
            print("Given OCRopy venv does not have a Python binary!")
            exit(1)

        if not os.access(python_path, os.X_OK):
            print("Given OCRopy venv Python binary is not executable!")
            exit(1)

        if not (ocropy_venv / "bin" / "ocropus-nlbin").is_file():
            print("Given OCRopy venv does not have an ocropus-nlbin script!")
            exit(1)

    if args.model_path is None:
        model_path = None
        if args.segmenter == "i6":
            print(
                "A model path must be specified when using " +
                "segmentation-pytorch!!"
            )
            exit(1)
    else:
        model_path = Path(args.model_path)
        if not model_path.is_file():
            print("Given model path does not point a file!")
            exit(1)

    facsimile_path = Path(args.facsimile_dir)
    if not facsimile_path.is_dir():
        print("Given facsimile directory is not a directory!")
        exit(1)

    progress_path = Path(args.progress_file)
    if progress_path.exists() and not progress_path.is_file():
        print("Given progress file is not a file!")
        exit(1)

    with sqlite3.connect(str(progress_path)) as conn:
        create_progress_schema(conn)
        schedule_downloaded_facsimiles(conn)

        sched = list(scheduled_facsimiles(conn))
        while True:
            if sched:
                segment_facsimiles(
                    conn,
                    facsimile_path, args.process_count, ocropy_venv,
                    args.segmenter, model_path, sched
                )
            else:
                print("Nothing to do. Waiting for work...")
                # Pause the thread so we don't waste CPU cycles waiting
                time.sleep(10.0)

            schedule_downloaded_facsimiles(conn)
            sched = list(scheduled_facsimiles(conn))


if __name__ == "__main__":
    main()
