import argparse
from datetime import datetime
import multiprocessing
import subprocess
import os
import itertools
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, TypeVar
from PIL import Image
from kraken import pageseg
from page.elements import PcGts, Page, Metadata, Region, Line, TextRegion
from dta_ocr import BoundingBox
from lxml import etree

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


def binarise_facsimile_chunk(
    ocropy_venv: Path, chunk_paths: List[Path], process_count: int
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
            ((ocropy_venv, chunk, process_count) for chunk in chunked(
                (fac for fac, _ in facsimiles(facsimile_path)),
                facsimile_count // process_count
            ))
        )


# computes regions in reading order
def segmentation_to_regions(seg: Dict[str, Any]) -> List[Region]:
    line_id = 0
    regions = []

    for xmin, ymin, xmax, ymax in seg["boxes"]:
        bbox = BoundingBox(xmin, ymin, xmax, ymax)
        coords = bbox.to_coords()
        line = Line(f"l{line_id}", coords, None, None)
        regions.append(TextRegion(f"r{line_id}", coords, [], None, [line]))
        line_id += 1

    return regions


def segment_facsimile(
    inp: Tuple[Path, Path]
):
    fac_path, parent_dir = inp
    fac_name = fac_path.stem

    bin_path = parent_dir / "bin" / f"{fac_name}.bin.png"
    if not bin_path.is_file():
        print(f"Could not find binarisation for {fac_path}!!")
        return

    bin_img = Image.open(bin_path)
    segmentation = pageseg.segment(bin_img)

    now = datetime.now()
    metadata = Metadata(
        "segment_facsimiles.py", now, now,
        "Generated using Kraken legacy segmentation."
    )
    page = Page(
        (bin_img.width, bin_img.height), fac_path.name,
        segmentation_to_regions(segmentation)
    )
    pcgts = PcGts(None, metadata, page)
    pcgts_xml = pcgts.to_element({})

    pagexml_path = parent_dir / f"{fac_name}.seg.xml"
    with pagexml_path.open("wb") as file:
        print(f"Writing {pagexml_path}...")
        file.write(etree.tostring(pcgts_xml, pretty_print=True))


def segment_facsimiles(
    facsimile_path: Path, process_count: int
):
    with multiprocessing.Pool(process_count) as pool:
        pool.map(segment_facsimile, facsimiles(facsimile_path))


def preprocess_facsimiles(
    facsimile_path: Path, process_count: int, ocropy_venv: Path
):
    # binarise_facsimiles(ocropy_venv, facsimile_path, process_count)
    segment_facsimiles(facsimile_path, process_count)


def main():
    cpu_count = multiprocessing.cpu_count()

    arg_parser = argparse.ArgumentParser(
        "segment_facsimiles",
        description=(
            "Given a set of facsimiles, in the same format as the output " +
            "of download_facsimiles.py, preprocesses (binarisation, " +
            "deskewing) and segments all images."
        )
    )
    arg_parser.add_argument(
        "--ocropy-venv", dest="ocropy_venv", required=True,
        help=(
            "Path to a virtualenv containing a Python 2.7 install along with" +
            "the OCRopy binaries."
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
            "download_facsimiles.py."
        )
    )
    args = arg_parser.parse_args()

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

    facsimile_path = Path(args.facsimile_dir)
    if not facsimile_path.is_dir():
        print("Given facsimile directory is not a directory!")
        exit(1)

    preprocess_facsimiles(facsimile_path, args.process_count, ocropy_venv)


if __name__ == "__main__":
    main()
