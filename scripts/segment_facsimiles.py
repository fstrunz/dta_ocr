import argparse
import multiprocessing
from pathlib import Path
from typing import Iterable, Tuple
from PIL import Image
from kraken import binarization


def facsimiles(facsimile_path: Path) -> Iterable[Tuple[Path, Path]]:
    for doc_path in facsimile_path.iterdir():
        if doc_path.is_dir():
            for fac_path in doc_path.iterdir():
                if fac_path.is_file():
                    yield fac_path, doc_path


def binarise_facsimile(inp: Tuple[Path, Path]):
    fac_path, parent_dir = inp

    print(f"Binarising {fac_path}...")
    bin_dir: Path = parent_dir / "bin"
    bin_dir.mkdir(exist_ok=True)

    # binarise image and save it in the bin/ dir
    # under the same name
    fac_img = Image.open(fac_path)
    bin_img = binarization.nlbin(fac_img)
    bin_img.save(bin_dir / fac_path.name)


def binarise_facsimiles(facsimile_path: Path, process_count: int):
    with multiprocessing.Pool(process_count) as pool:
        pool.map(binarise_facsimile, facsimiles(facsimile_path))


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

    facsimile_path = Path(args.facsimile_dir)
    if not facsimile_path.is_dir():
        print("Given facsimile directory is not a directory!")
        exit(1)

    binarise_facsimiles(facsimile_path, args.process_count)


if __name__ == "__main__":
    main()
