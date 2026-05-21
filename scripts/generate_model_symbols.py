from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REPORT_ONLY_SYMBOLS = {"SPY", "DIA", "IWF", "QQQ", "EQAL", "RSP", "QQQE"}


def col_number(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    number = 0
    for char in letters.upper():
        number = number * 26 + ord(char) - 64
    return number


def clean_symbol(value: object) -> str:
    symbol = str(value or "").strip().upper()
    if not symbol or symbol in {"TICKER", "SYMBOL", "NAN", "NONE", "CASH"}:
        return ""
    if symbol in REPORT_ONLY_SYMBOLS:
        return ""
    if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", symbol):
        return ""
    return symbol


def read_xlsx_rows(path: Path, sheet_path: str = "xl/worksheets/sheet1.xml") -> list[list[str]]:
    with ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", NS):
                shared_strings.append("".join(text.text or "" for text in item.findall(".//m:t", NS)))

        root = ET.fromstring(archive.read(sheet_path))
        rows: list[list[str]] = []
        for row in root.findall(".//m:sheetData/m:row", NS):
            values: dict[int, str] = {}
            for cell in row.findall("m:c", NS):
                ref = cell.attrib.get("r", "")
                cell_type = cell.attrib.get("t")
                raw_value = cell.find("m:v", NS)
                value = ""
                if cell_type == "inlineStr":
                    value = "".join(text.text or "" for text in cell.findall(".//m:t", NS))
                elif raw_value is not None:
                    value = raw_value.text or ""
                    if cell_type == "s" and value:
                        value = shared_strings[int(value)] if int(value) < len(shared_strings) else ""
                if ref:
                    values[col_number(ref)] = value

            if values:
                rows.append([values.get(idx, "") for idx in range(1, max(values) + 1)])
        return rows


def extract_symbols(path: Path, column_index: int, start_row_index: int = 0) -> list[str]:
    rows = read_xlsx_rows(path)
    symbols: list[str] = []
    seen: set[str] = set()
    for row in rows[start_row_index:]:
        value = row[column_index] if column_index < len(row) else ""
        symbol = clean_symbol(value)
        if symbol and symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
    return symbols


def main() -> int:
    if len(sys.argv) != 4:
        print("Usage: generate_model_symbols.py <allocations_dir> <output_json> <output_csv>", file=sys.stderr)
        return 2

    allocations_dir = Path(sys.argv[1])
    output_json = Path(sys.argv[2])
    output_csv = Path(sys.argv[3])

    model_symbols = {
        "GRIP": extract_symbols(allocations_dir / "grip_allocation.xlsx", column_index=3, start_row_index=2),
        "EDIP": extract_symbols(allocations_dir / "DSIP allocation.xlsx", column_index=1, start_row_index=0),
    }
    model_symbols["ALL SYMBOLS"] = sorted(set(model_symbols["GRIP"]) | set(model_symbols["EDIP"]))

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(model_symbols, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    output_csv.write_text(
        "model,symbol\n"
        + "".join(f"{model},{symbol}\n" for model, symbols in model_symbols.items() for symbol in symbols),
        encoding="utf-8",
    )
    print(f"Wrote {output_json} and {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
