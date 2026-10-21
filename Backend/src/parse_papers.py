"""
Parse PMC / JATS XML articles into the JSON schema used by the medical RAG pipeline.

Usage:
    python parse_papers.py

    from parse_papers import Parser
    parser = Parser(xml_subfolder="data/raw/lyme_disease")
    papers = parser.parse_all()       # list of paper dicts, nothing written
    parser.save(papers)               # -> lyme_disease_001.json, lyme_disease_002.json, ...
    parser.run()                      # parse_all() + save() in one call

Paper IDs (and JSON file names) are named after the folder the XML files live
in: data/raw/lyme_disease -> lyme_disease_001, lyme_disease_002, ...
Pass id_prefix="..." to override. If the XML files sit directly in the project
root, the prefix falls back to "paper".

Input is always inside the project root, ClinicalClub/ (this file lives in
ClinicalClub/Backend/src, so the root is two folders up). Standard library only, so nothing to compile or install.
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

# parse_papers.py lives in ClinicalClub/Backend/src -> parents[2] is ClinicalClub.
# If you move this file, change the number to match how deep it sits.
PROJECT_DIR = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------ text helpers ---

def _clean(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _text(el: Optional[ET.Element]) -> str:
    """All text inside an element, whitespace-normalised."""
    return _clean(" ".join(el.itertext())) if el is not None else ""


def _tidy_punct(s: str) -> str:
    # itertext joins with spaces, which leaves "word ." / "( 1 )" artifacts
    s = re.sub(r"\s+([.,;:!?)\]])", r"\1", s)
    return re.sub(r"([(\[])\s+", r"\1", s)


def _initials(given: Optional[str]) -> str:
    """'John' -> 'J', 'Aisha R' -> 'AR', 'MK' (already initials) -> 'MK'."""
    given = _clean(given)
    compact = given.replace(" ", "").replace(".", "")
    if re.fullmatch(r"[A-Z]{1,4}", compact):
        return compact
    return "".join(p[0].upper() for p in re.split(r"[\s\-.]+", given) if p)


def _slug(name: str) -> str:
    """'Lyme Disease' -> 'lyme_disease'; safe for use in IDs and file names."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _strip_namespaces(root: ET.Element) -> ET.Element:
    """Drop namespace prefixes from tags and attributes ({ns}href -> href).
    PMC's OAI service wraps articles in a default namespace; this makes
    every lookup work the same either way."""
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
        for key in list(el.attrib):
            if "}" in key:
                el.attrib[key.split("}", 1)[1]] = el.attrib.pop(key)
    return root


# ------------------------------------------------------------------ Parser ---

class Parser:
    """Turns the project's JATS/PMC XML papers into paper dicts / JSON files."""

    # Element types that count as body text directly inside a <sec>.
    # Figures, tables, equations and sub-sections are deliberately not here.
    TEXT_BLOCKS = {"p", "list", "disp-quote", "boxed-text", "def-list", "statement"}

    LICENSE_PATTERNS = [
        (r"publicdomain/zero", "CC0 1.0"),
        (r"by-nc-nd/(\d\.\d)", "CC BY-NC-ND {}"),
        (r"by-nc-sa/(\d\.\d)", "CC BY-NC-SA {}"),
        (r"by-nc/(\d\.\d)", "CC BY-NC {}"),
        (r"by-nd/(\d\.\d)", "CC BY-ND {}"),
        (r"by-sa/(\d\.\d)", "CC BY-SA {}"),
        (r"licenses/by/(\d\.\d)", "CC BY {}"),
    ]

    def __init__(self, xml_subfolder: str = "", output_subfolder: str = "json_papers",
                 project_dir: Path = PROJECT_DIR, id_prefix: Optional[str] = None):
        self.project_dir = Path(project_dir).resolve()
        self.xml_dir = (self.project_dir / xml_subfolder).resolve()
        self.output_dir = (self.project_dir / output_subfolder).resolve()
        if not self.xml_dir.is_relative_to(self.project_dir):
            raise ValueError(f"xml_subfolder must be inside {self.project_dir}")
        # IDs are named after the XML folder (lyme_disease -> lyme_disease_001)
        # unless a prefix is given; XML in the project root falls back to "paper".
        default = self.xml_dir.name if self.xml_dir != self.project_dir else "paper"
        self.id_prefix = _slug(id_prefix or default) or "paper"
        self.skipped: list[tuple[str, str]] = []   # (filename, reason) from the last parse_all()

    # ------------------------------------------------------------ public API

    def xml_files(self) -> list[Path]:
        """The XML files this parser will read, in a stable (sorted) order."""
        return sorted(self.xml_dir.glob("*.xml"))

    def parse_file(self, xml_file: str | Path, paper_id: Optional[str] = None) -> dict:
        """Parse one project XML file (a name like 'PMC123.xml' or a path inside
        the XML folder) into a paper dict. Raises ValueError if it isn't a paper."""
        path = (self.xml_dir / xml_file).resolve()
        if not path.is_relative_to(self.xml_dir):
            raise ValueError(f"{xml_file} is not in the project XML folder {self.xml_dir}")

        root = _strip_namespaces(ET.parse(path).getroot())
        article = root if root.tag == "article" else root.find(".//article")
        if article is None:
            raise ValueError("no <article> element (not a JATS paper)")
        front = article.find("front")
        meta = front.find("article-meta") if front is not None else None
        if meta is None:
            raise ValueError("no <article-meta>")

        ids = self._article_ids(meta)
        pmcid = ids.get("pmcid") or ids.get("pmc") or ids.get("pmcaid", "")
        if pmcid and not pmcid.upper().startswith("PMC"):
            pmcid = f"PMC{pmcid}"

        return {
            "paper_id": paper_id or pmcid or path.stem,
            "title": _tidy_punct(_text(meta.find("title-group/article-title"))),
            "authors": self._authors(meta),
            "year": self._year(meta),
            "journal": self._journal(front.find("journal-meta")),
            "doi": ids.get("doi", ""),
            "pmcid": pmcid,
            "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/" if pmcid else "",
            "license": self._license(meta),
            "abstract": self._abstract(meta),
            "sections": self._sections(article),
            "references": self._references(article),
        }

    def parse_all(self) -> list[dict]:
        """Parse every project XML file. Files that aren't valid papers are
        skipped and recorded in self.skipped; paper_ids have no gaps and are
        named after the XML folder (e.g. lyme_disease_001)."""
        files = self.xml_files()
        if not files:
            raise FileNotFoundError(f"No .xml files found in {self.xml_dir}")
        papers, self.skipped = [], []
        for f in files:
            try:
                paper_id = f"{self.id_prefix}_{len(papers) + 1:03d}"
                papers.append(self.parse_file(f.name, paper_id=paper_id))
            except (ET.ParseError, ValueError) as e:
                self.skipped.append((f.name, str(e)))
                print(f"[skip] {f.name}: {e}", file=sys.stderr)
        return papers

    def save(self, papers: list[dict]) -> list[Path]:
        """Write each paper to output_dir/<paper_id>.json."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for paper in papers:
            out = self.output_dir / f"{paper['paper_id']}.json"
            out.write_text(json.dumps(paper, indent=2, ensure_ascii=False), encoding="utf-8")
            written.append(out)
        return written

    def run(self) -> list[Path]:
        """Parse all project XML files and write them as JSON."""
        return self.save(self.parse_all())

    # ------------------------------------------------------------ metadata

    @staticmethod
    def _article_ids(meta: ET.Element) -> dict:
        return {aid.get("pub-id-type", ""): _clean(aid.text) for aid in meta.findall("article-id")}

    @staticmethod
    def _authors(meta: ET.Element) -> list[str]:
        """'Smith J' style: surname + initials. Editors etc. are excluded."""
        out = []
        for contrib in meta.findall(".//contrib-group/contrib"):
            if contrib.get("contrib-type", "author") != "author":
                continue
            name = contrib.find("name")
            if name is not None:
                out.append(f"{_clean(name.findtext('surname'))} "
                           f"{_initials(name.findtext('given-names'))}".strip())
            elif contrib.find("collab") is not None:          # group authors
                out.append(_text(contrib.find("collab")))
        return [a for a in out if a]

    @staticmethod
    def _year(meta: ET.Element) -> Optional[int]:
        # Prefer the actual publication date over collection/release dates.
        preferred = ("epub", "ppub", "pub", "collection", "pmc-release")

        def rank(d):
            kind = d.get("pub-type") or d.get("date-type") or ""
            return preferred.index(kind) if kind in preferred else 99

        for d in sorted(meta.findall("pub-date"), key=rank):
            y = _clean(d.findtext("year"))
            if y.isdigit():
                return int(y)
        return None

    @staticmethod
    def _journal(journal_meta: Optional[ET.Element]) -> str:
        if journal_meta is None:
            return ""
        return (_text(journal_meta.find(".//journal-title"))
                or _text(journal_meta.find("journal-id[@journal-id-type='nlm-ta']")))

    def _license(self, meta: ET.Element) -> str:
        lic = meta.find(".//permissions/license")
        if lic is None:
            return ""
        # URL lives in xlink:href on <license> or in <ali:license_ref>
        url = lic.get("href") or _clean(lic.findtext("license_ref"))
        for pat, label in self.LICENSE_PATTERNS:
            m = re.search(pat, url or "")
            if m:
                return label.format(*m.groups())
        if url:
            return url
        blob = _text(lic)  # fall back to the human-readable license text
        m = re.search(r"CC[\s-]?BY(?:[\s-](?:NC|ND|SA))*(?:\s*\d\.\d)?", blob, re.I)
        return m.group(0).upper().replace("-", " ") if m else blob[:200]

    @staticmethod
    def _abstract(meta: ET.Element) -> str:
        """Main abstract (graphical/teaser variants skipped). Structured
        abstracts are flattened as 'Background: ... Methods: ...'."""
        abstracts = meta.findall("abstract")
        main = next((a for a in abstracts if not a.get("abstract-type")),
                    abstracts[0] if abstracts else None)
        if main is None:
            return ""
        secs = main.findall("sec")
        if not secs:
            return _tidy_punct(_text(main))
        parts = []
        for s in secs:
            title = _text(s.find("title"))
            body = " ".join(_text(p) for p in s.findall("p"))
            parts.append(f"{title}: {body}" if title else body)
        return _tidy_punct(" ".join(parts))

    # ------------------------------------------------------------ sections

    def _section_body(self, sec: ET.Element) -> str:
        """Only this section's own text; sub-sections become their own entries."""
        chunks = []
        for child in sec:
            if child.tag not in self.TEXT_BLOCKS:
                continue
            if child.tag == "list":
                chunks.append("; ".join(_text(li) for li in child.findall("list-item")))
            else:
                chunks.append(_text(child))
        return _tidy_punct(" ".join(c for c in chunks if c))

    def _walk_sections(self, sec: ET.Element, parent: str, out: list) -> None:
        heading = _text(sec.find("title"))
        text = self._section_body(sec)
        if heading or text:
            out.append({"heading": heading, "parent": parent, "text": text})
        for child in sec.findall("sec"):
            # untitled sub-sections inherit the nearest titled ancestor as parent
            self._walk_sections(child, heading or parent, out)

    def _sections(self, article: ET.Element) -> list[dict]:
        body = article.find("body")
        if body is None:
            return []
        out: list[dict] = []
        loose = self._section_body(body)   # paragraphs placed directly in <body>
        if loose:
            out.append({"heading": "", "parent": "", "text": loose})
        for sec in body.findall("sec"):
            self._walk_sections(sec, "", out)
        return out

    # ---------------------------------------------------------- references

    @staticmethod
    def _format_element_citation(cit: ET.Element) -> str:
        """Readable 'Authors. Title. Journal. Year;Vol(Issue):pages.' string."""
        names = [f"{_clean(n.findtext('surname'))} {_initials(n.findtext('given-names'))}".strip()
                 for n in (cit.findall(".//person-group/name") or cit.findall("name"))]
        names += [_text(c) for c in cit.findall(".//collab")]
        parts = []
        if names:
            parts.append(", ".join(names) + (", et al" if cit.find(".//etal") is not None else ""))
        parts += [t for t in (_text(cit.find(tag)) for tag in
                              ("article-title", "chapter-title", "source")) if t]
        tail = _clean(cit.findtext("year"))
        vol, issue = _clean(cit.findtext("volume")), _clean(cit.findtext("issue"))
        fpage, lpage = _clean(cit.findtext("fpage")), _clean(cit.findtext("lpage"))
        if vol:
            tail += f";{vol}" + (f"({issue})" if issue else "")
        if fpage:
            tail += f":{fpage}" + (f"-{lpage}" if lpage else "")
        if tail:
            parts.append(tail)
        return ". ".join(p.rstrip(".") for p in parts) + "."

    def _references(self, article: ET.Element) -> list[dict]:
        refs = []
        for ref in article.findall("back/ref-list//ref"):
            cit = next((c for tag in ("mixed-citation", "element-citation",
                                      "citation", "nlm-citation")
                        if (c := ref.find(tag)) is not None), None)
            if cit is None:
                continue
            doi = next((_clean(p.text) for p in cit.iter("pub-id")
                        if p.get("pub-id-type") == "doi"), "")
            if not doi:   # some citations embed the DOI as plain text
                m = re.search(r"\b10\.\d{4,9}/\S+[^\s.,;]", _text(cit))
                doi = m.group(0) if m else ""
            if cit.tag == "mixed-citation":
                for pid in cit.iter("pub-id"):   # keep DOIs/PMIDs out of the text
                    pid.text = ""
                text = _tidy_punct(_text(cit)).rstrip(" .") + "."
            else:
                text = self._format_element_citation(cit)
            refs.append({"ref_id": ref.get("id", f"r{len(refs) + 1}"),
                         "text": text, "doi": doi})
        return refs


def run_all_diseases(only: Optional[list[str]] = None,
                     raw_subfolder: str = "data/raw",
                     processed_subfolder: str = "data/processed") -> dict[str, list[Path]]:
    """Parse every disease folder under data/raw/ into data/processed/<disease>/.
    Pass `only` (folder names) to limit it to specific diseases."""
    raw_root = PROJECT_DIR / raw_subfolder
    folders = sorted(p for p in raw_root.iterdir() if p.is_dir())
    if only:
        wanted = set(only)
        missing = wanted - {f.name for f in folders}
        for name in sorted(missing):
            print(f"[skip] {name}: no such folder in {raw_root}", file=sys.stderr)
        folders = [f for f in folders if f.name in wanted]

    results: dict[str, list[Path]] = {}
    for folder in folders:
        parser = Parser(xml_subfolder=f"{raw_subfolder}/{folder.name}",
                        output_subfolder=f"{processed_subfolder}/{folder.name}")
        try:
            files = parser.run()
        except FileNotFoundError:
            print(f"[skip] {folder.name}: no XML files", file=sys.stderr)
            continue
        results[folder.name] = files
        print(f"{folder.name}: wrote {len(files)} JSON files"
              + (f" ({len(parser.skipped)} skipped)" if parser.skipped else ""))
    print(f"Done: {sum(len(v) for v in results.values())} papers "
          f"across {len(results)} disease folder(s).")
    return results


if __name__ == "__main__":
    # python parse_papers.py                    -> every folder in data/raw/
    # python parse_papers.py lyme_disease asthma -> just those folders
    run_all_diseases(only=sys.argv[1:] or None)