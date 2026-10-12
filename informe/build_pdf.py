"""Convierte informe_final.md a HTML paginado y luego a PDF con Chrome headless."""
import base64
import mimetypes
import re
import subprocess
from pathlib import Path

import markdown

BASE = Path(__file__).parent
MD = BASE / "informe_final.md"
HTML = BASE / "informe_final.html"
PDF = BASE / "informe_final.pdf"
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

CSS = """
@page { size: A4; margin: 21mm 19mm; }
body {
  font-family: "Segoe UI", Calibri, sans-serif;
  font-size: 10.6pt; line-height: 1.58; color: #1a1a1a;
  text-align: justify; hyphens: auto;
}
h1 {
  font-size: 20pt; color: #0f3d6b; text-align: left;
  margin: 0 0 4pt; line-height: 1.22; border-bottom: 2.5pt solid #0f3d6b;
  padding-bottom: 6pt;
}
h1 + h3 {
  font-size: 11.5pt; color: #4a5568; font-weight: 500;
  margin: 0 0 14pt; border: none; text-align: left;
}
h2 {
  font-size: 13.2pt; color: #0f3d6b; margin: 17pt 0 7pt;
  border-bottom: 0.7pt solid #c5d2e0; padding-bottom: 3pt;
  page-break-after: avoid; text-align: left;
}
h3 {
  font-size: 11.1pt; color: #1e4d7b; margin: 12pt 0 5pt;
  page-break-after: avoid; text-align: left;
}
h4 { font-size: 10.3pt; margin: 9pt 0 4pt; page-break-after: avoid; }
p { margin: 0 0 6.5pt; orphans: 3; widows: 3; }
strong { color: #0d2d4f; }
table {
  border-collapse: collapse; width: 100%;
  margin: 9pt 0 12pt; font-size: 9.3pt;
  page-break-inside: avoid;
}
th {
  background: #0f3d6b; color: #fff; font-weight: 600;
  padding: 6pt 7pt; text-align: left; border: 0.5pt solid #0f3d6b;
}
td { padding: 5.2pt 7pt; border: 0.5pt solid #ccd6e0; vertical-align: top; }
tbody tr:nth-child(even) { background: #f4f7fa; }
code {
  font-family: Consolas, monospace; font-size: 8.6pt;
  background: #eef2f6; padding: 1pt 3.5pt; border-radius: 2pt;
  color: #9b2226;
}
pre {
  background: #f7f9fb; border: 0.5pt solid #d4dde6;
  border-left: 2.5pt solid #0f3d6b; border-radius: 2pt;
  padding: 7pt 9pt; margin: 7pt 0 10pt; overflow-x: auto;
  page-break-inside: avoid;
}
pre code {
  background: none; padding: 0; color: #1a1a1a;
  font-size: 8.5pt; line-height: 1.42;
}
blockquote {
  margin: 9pt 0; padding: 7pt 11pt;
  background: #fdf6e3; border-left: 3pt solid #d9a441;
  font-size: 9.5pt;
}
blockquote p { margin: 0; }
ul, ol { margin: 0 0 7pt; padding-left: 17pt; }
li { margin-bottom: 3.2pt; }
img {
  max-width: 78%; max-height: 62mm; height: auto;
  display: block; margin: 7pt auto 4pt;
  border: 0.5pt solid #ccd6e0; object-fit: contain;
  page-break-inside: avoid;
}
hr {
  border: none; border-top: 0.6pt solid #d4dde6;
  margin: 15pt 0; page-break-after: avoid;
}
"""


def incrustar_imagenes(html: str) -> str:
    """Convierte src locales a data URI para que Chrome headless las resuelva."""
    def _rep(m):
        src = m.group(1)
        ruta = (BASE / src).resolve()
        if not ruta.exists():
            print(f"  aviso: no se encontro {src}")
            return m.group(0)
        mime = mimetypes.guess_type(ruta.name)[0] or "image/png"
        b64 = base64.b64encode(ruta.read_bytes()).decode()
        return f'src="data:{mime};base64,{b64}"'

    return re.sub(r'src="([^"]+)"', _rep, html)


def main():
    cuerpo = markdown.markdown(
        MD.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "attr_list"],
    )
    cuerpo = incrustar_imagenes(cuerpo)

    HTML.write_text(
        f"<!DOCTYPE html><html lang='es'><head><meta charset='utf-8'>"
        f"<title>Informe TP - Vision por Computadora</title>"
        f"<style>{CSS}</style></head><body>{cuerpo}</body></html>",
        encoding="utf-8",
    )
    print(f"HTML generado: {HTML.name} ({HTML.stat().st_size / 1024:.0f} KB)")

    if not CHROME.exists():
        print("Chrome no encontrado: abri el HTML y usa Imprimir > Guardar como PDF")
        return

    subprocess.run(
        [str(CHROME), "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={PDF}", HTML.as_uri()],
        check=True, capture_output=True, timeout=180,
    )
    print(f"PDF generado: {PDF.name} ({PDF.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
