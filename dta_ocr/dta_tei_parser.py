import re
from dataclasses import dataclass
from typing import Dict, Optional, Union
from bs4 import BeautifulSoup, Tag, NavigableString

PB_REGEX = re.compile(r"#f0*(\d+)")


@dataclass
class DTAPage:
    facsimile: int
    text: str


@dataclass
class DTADocument:
    pages: Dict[int, DTAPage]

    def get_page_text(self, facsimile: int) -> str:
        return self.pages[facsimile].text

    @staticmethod
    def from_tei_soup(soup: BeautifulSoup) -> Optional["DTADocument"]:
        return TEIParser(soup).parse()


class TEIParser:
    def __init__(self, soup: BeautifulSoup):
        self.soup = soup

    # Given the text tag from a TEI document, recur through the
    # children and populate the pages dictionary.
    def __find_text(
        self, node: Union[Tag, NavigableString], pages: Dict[int, DTAPage],
        facs: Optional[int] = None
    ) -> int:
        if isinstance(node, Tag):
            if node.name == "pb":
                facs = int(PB_REGEX.match(node["facs"]).group(1))
                pages[facs] = DTAPage(facs, "")
            else:
                for child in node:
                    facs = self.__find_text(child, pages, facs)
        elif isinstance(node, NavigableString) and facs is not None:
            pages[facs].text += f"{str(node.strip())} "

        return facs

    def parse(self) -> DTADocument:
        text = self.soup.find("text")
        if text is None:
            raise ValueError("TEI file does not have a text tag")

        pages = {}
        self.__find_text(text, pages)
        return DTADocument(pages)
