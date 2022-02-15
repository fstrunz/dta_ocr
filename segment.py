import argparse
from datetime import datetime
import multiprocessing
import subprocess
import os
import itertools
import torch
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
    inp: Tuple[Path, Path]
):
    fac_path, parent_dir = inp
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
    pcgts.save_to_file(parent_dir / "bin" / f"{fac_name}.xml")


def segment_facsimiles_kraken(
    facsimile_path: Path
):
    for fac_path in facsimiles(facsimile_path):
        segment_facsimile_kraken(fac_path)


def segment_facsimile_i6(
    inp: Tuple[Path, Path], predictor: Predictor, pool: multiprocessing.Pool
):
    fac_path, parent_dir = inp
    print(f"Segmenting {fac_path}...")

    img = SourceImage.load(fac_path)
    prediction, scaled_image = predictor.predict_image(img, process_pool=pool)
    layout_settings = LayoutProcessingSettings(
        marginalia_postprocessing=False,
        source_scale=True,
        layout_method=LayoutProcessingMethod.LINES_ONLY
    )

    analysed_content = process_layout(
        prediction, scaled_image, pool, layout_settings
    )
    analysed_content.to_pagexml_space(prediction.prediction_scale_factor)
    xml_gen = analysed_content.export(
        scaled_image, fac_path, simplified_xml=False
    )

    output_path = parent_dir / "bin"
    xml_gen.save_textregions_as_xml(str(output_path.absolute()))


def segment_facsimiles_i6(
    facsimile_path: Path, model_path: Path, process_count: int
):
    print("Setting up predictor...")
    settings = PredictionSettings([model_path], 1000000, None, None)
    predictor = Predictor(settings)

    if not torch.cuda.is_available():
        torch.set_num_threads(process_count)

    with multiprocessing.Pool(process_count) as pool:
        for fac_path in facsimiles(facsimile_path):
            segment_facsimile_i6(fac_path, predictor, pool)


def preprocess_facsimiles(
    facsimile_path: Path, process_count: int, ocropy_venv: Optional[Path],
    segmenter: str, model_path: Path
):
    print("Segmenting facsimiles...")

    if segmenter == "kraken":
        print("Binarising facsimiles with OCRopus...")
        binarise_facsimiles(ocropy_venv, facsimile_path, process_count)
        segment_facsimiles_kraken(facsimile_path, process_count)
    elif segmenter == "i6":
        # This segmenter automatically does binarisation
        segment_facsimiles_i6(facsimile_path, model_path, process_count)


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
            "Path to a virtualenv containing a Python 2.7 install along with" +
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

    preprocess_facsimiles(
        facsimile_path, args.process_count, ocropy_venv,
        args.segmenter, model_path
    )


if __name__ == "__main__":
    main()
