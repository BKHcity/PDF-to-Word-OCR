"""Command-line interface for the `handocr` package.

Usage examples:
  python -m handocr.cli predict --image path/to/img.png
  python -m handocr.cli export-pdf --input doc.pdf --out out_searchable.pdf
"""
import argparse
import json
from pathlib import Path
from PIL import Image

from . import predict_image, export_searchable_pdf


def cmd_predict(args):
    path = args.image
    res = predict_image(path, char_level=not args.disable_char_level)
    out = {
        "text": res.text,
        "source": res.source,
        "success": res.success,
    }
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print("=== Recognized Text ===")
        print(res.text)


def cmd_export_pdf(args):
    inp = args.input
    outp = args.out
    success, msg = export_searchable_pdf(inp, outp, dpi=args.dpi)
    if success:
        print(f"Searchable PDF saved to: {outp}")
    else:
        print(f"Failed: {msg}")


def main():
    parser = argparse.ArgumentParser(prog="handocr")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("predict", help="Recognize text in an image")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--json", action="store_true", help="Print output as JSON")
    p.add_argument("--disable-char-level", action="store_true", help="Disable char-level fusion")
    p.set_defaults(func=cmd_predict)

    e = sub.add_parser("export-pdf", help="Export searchable PDF from input PDF/image")
    e.add_argument("--input", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--dpi", type=int, default=150)
    e.set_defaults(func=cmd_export_pdf)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
