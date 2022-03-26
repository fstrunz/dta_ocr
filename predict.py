import argparse
import sqlite3
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import List
from pathlib import Path
from calamari_ocr.ocr.predict.predictor import MultiPredictor
from calamari_ocr.ocr.dataset.datareader.pagexml.reader import (
    PageXML, PageXMLReader
)
from calamari_ocr.ocr.dataset.pipeline import CalamariPipeline
from dta_ocr import create_progress_schema
from lxml import etree


def create_predictor(model_path: Path) -> MultiPredictor:
    checkpoints: List[str] = [
        str(model_path / model_file.stem)
        for model_file in model_path.iterdir()
        if model_file.is_file() and model_file.suffix == ".json"
    ]

    return MultiPredictor.from_paths(checkpoints)


def schedule_segmented_documents(conn: sqlite3.Connection):
    print("Scheduling segmented documents...")
    seg_cursor = conn.execute(
        """SELECT dta_dirname, page_number FROM segmentations
        WHERE status = 'finished'
        ORDER BY dta_dirname, page_number"""
    )

    for dta_dirname, page_number in seg_cursor:
        conn.execute(
            """INSERT OR IGNORE INTO predictions (
                dta_dirname, page_number, prediction_path, status
            ) VALUES ( ?, ?, NULL, 'pending' )""",
            (dta_dirname, page_number)
        )
        conn.commit()

    seg_cursor.close()


class Typeface(Enum):
    Antiqua = 0
    Fraktur = 1


@dataclass
class Prediction:
    dta_dirname: str
    page_number: int
    tei_path: Path
    seg_path: Path
    typeface: Typeface = field(init=False)

    def __post_init__(self):
        try:
            with self.tei_path.open("r", encoding="utf8") as tei_file:
                tree = etree.parse(tei_file)

            root_xml: etree.ElementBase = tree.getroot()
            p_xml = root_xml.find(
                ".//physDesc/typeDesc/p", namespaces=root_xml.nsmap
            )
            if "Antiqua" in p_xml.text:
                self.typeface = Typeface.Antiqua
            else:
                self.typeface = Typeface.Fraktur
        except Exception:
            self.typeface = None
            raise RuntimeError(
                f"Failed to determine typeface for {str(self.tei_path)}!!"
            )


def fetch_scheduled_predictions(conn: sqlite3.Connection) -> List[Prediction]:
    print("Fetching scheduled predictions...")
    pred_cursor = conn.execute(
        """SELECT pred.dta_dirname, pred.page_number, doc.tei_path, seg.file_path
        FROM predictions pred
        NATURAL JOIN documents doc
        JOIN segmentations seg
        ON seg.dta_dirname = pred.dta_dirname
        AND seg.page_number = pred.page_number
        WHERE pred.status != 'finished'"""
    )

    return [
        Prediction(dta_dirname, page_number, Path(tei_path), Path(seg_path))
        for dta_dirname, page_number, tei_path, seg_path in pred_cursor
    ]


def predict(
    facsimile_path: Path, antiqua_path: Path,
    fraktur_path: Path, scheduled: List[Prediction]
) -> List[Path]:
    antiqua_pred: MultiPredictor = create_predictor(antiqua_path)
    fraktur_pred: MultiPredictor = create_predictor(fraktur_path)

    xml_files_f = [
        str(sched.seg_path) for sched in scheduled
        if sched.typeface == Typeface.Fraktur
    ]
    images_f = [
        str(facsimile_path / sched.dta_dirname / f"{sched.page_number}.jpg")
        for sched in scheduled if sched.typeface == Typeface.Fraktur
    ]

    data_f = PageXML(
        images=images_f,
        xml_files=xml_files_f,
    )

    xml_files_a = [
        str(sched.seg_path) for sched in scheduled
        if sched.typeface == Typeface.Antiqua
    ]
    images_a = [
        str(facsimile_path / sched.dta_dirname / f"{sched.page_number}.jpg")
        for sched in scheduled if sched.typeface == Typeface.Antiqua
    ]

    data_a = PageXML(
        images=images_a,
        xml_files=xml_files_a
    )

    do_pred_f = fraktur_pred.predict(data_f)
    do_pred_a = antiqua_pred.predict(data_a)

    pipeline_f: CalamariPipeline = fraktur_pred.data.get_or_create_pipeline(
        fraktur_pred.params.pipeline, data_f
    )
    pipeline_a: CalamariPipeline = antiqua_pred.data.get_or_create_pipeline(
        antiqua_pred.params.pipeline, data_a
    )
    reader_f: PageXMLReader = pipeline_f.reader()
    reader_a: PageXMLReader = pipeline_a.reader()

    if len(reader_f) > 0:
        for s in do_pred_f:
            _, (_, pred), meta = s.inputs, s.outputs, s.meta
            reader_f.store_text_prediction(pred, meta["id"], None)

        reader_f.store()

    if len(reader_a) > 0:
        for s in do_pred_a:
            _, (_, pred), meta = s.inputs, s.outputs, s.meta
            reader_a.store_text_prediction(pred, meta["id"], None)

        reader_a.store()


def main():
    arg_parser = argparse.ArgumentParser(
        "predict",
        description=(
            "Given a set of facsimiles with segmentations, in the same " +
            "format as the output of segment.py, Calamari is used to " +
            "predict the contents of the generated segmentations."
        )
    )
    arg_parser.add_argument(
        "--facsimile-dir", dest="facsimile_dir", default="facsimiles",
        help=(
            "The facsimile directory as generated by download.py " +
            "and segment.py."
        )
    )
    arg_parser.add_argument(
        "--antiqua-dir", dest="antiqua_dir",
        default="./models/prediction/antiqua",
        help="The path of the Antiqua model directory to use for prediction."
    )
    arg_parser.add_argument(
        "--fraktur-dir", dest="fraktur_dir",
        default="./models/prediction/fraktur",
        help="The path of the Fraktur model directory to use for prediction."
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
    facsimile_path = Path(args.facsimile_dir)
    if not facsimile_path.is_dir():
        print(
            f"The given facsimile path ({facsimile_path}) is not a directory!"
        )
        exit(1)

    antiqua_path = Path(args.antiqua_dir)
    if not antiqua_path.is_dir():
        print(
            f"The given Antiqua path ({antiqua_path}) is not a " +
            "directory!"
        )
        exit(1)

    fraktur_path = Path(args.fraktur_dir)
    if not fraktur_path.is_dir():
        print(
            f"The given Fraktur path ({fraktur_path}) is not a " +
            "directory!"
        )
        exit(1)

    with sqlite3.connect(args.progress_file) as conn:
        create_progress_schema(conn)
        schedule_segmented_documents(conn)

        sched = list(fetch_scheduled_predictions(conn))
        while True:
            if sched:
                predict(facsimile_path, antiqua_path, fraktur_path, sched)

                # Compute the corresponding output files that Calamari
                # has hopefully produced and write these into the database.
                conn.executemany(
                    """UPDATE predictions
                    SET status = 'finished',
                        prediction_path = ?
                    WHERE dta_dirname = ?
                    AND page_number = ?""",
                    (
                        (
                            str(
                                s.seg_path.parent /
                                f"{s.seg_path.stem}.pred.xml"
                            ),
                            s.dta_dirname,
                            s.page_number
                        ) for s in sched
                    )
                )
                conn.commit()
            else:
                print("No work to be done. Waiting for more segmentations...")
                time.sleep(10.0)

            schedule_segmented_documents(conn)
            sched = list(fetch_scheduled_predictions(conn))


if __name__ == "__main__":
    main()
