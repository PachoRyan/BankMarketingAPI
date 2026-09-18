"""
src/download_data.py
====================
Descarga el Bank Marketing Dataset (UCI) a `data/bank-additional-full.csv`.

El CSV no se versiona en Git (ver .gitignore), así que este script hace el
proyecto reproducible: cualquiera clona el repo, ejecuta

    python src/download_data.py

y ya puede entrenar con `python src/train.py`.

Se usa la versión `bank-additional-full.csv` (41.188 filas, 20 features), que es
la recomendada por los autores del dataset porque incluye los indicadores
macroeconómicos.
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DESTINO = ROOT / "data" / "bank-additional-full.csv"

URL_UCI = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
# Espejo en GitHub por si la descarga desde UCI falla (su servidor a veces
# rechaza conexiones o la red corporativa lo bloquea).
URL_ESPEJO = "https://raw.githubusercontent.com/selva86/datasets/master/bank-full.csv"

NOMBRE_CSV = "bank-additional-full.csv"


def _descargar(url: str) -> bytes:
    print(f"Descargando: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _extraer_csv_de_zip(contenido: bytes) -> bytes | None:
    """El zip de UCI contiene otro zip (bank-additional.zip) dentro."""
    with zipfile.ZipFile(io.BytesIO(contenido)) as zf:
        for nombre in zf.namelist():
            if nombre.endswith(NOMBRE_CSV):
                return zf.read(nombre)
        for nombre in zf.namelist():
            if nombre.endswith(".zip"):
                interno = _extraer_csv_de_zip(zf.read(nombre))
                if interno:
                    return interno
    return None


def main() -> int:
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    if DESTINO.exists():
        print(f"El archivo ya existe: {DESTINO} ({DESTINO.stat().st_size / 1e6:.1f} MB)")
        return 0

    csv_bytes: bytes | None = None
    try:
        csv_bytes = _extraer_csv_de_zip(_descargar(URL_UCI))
    except Exception as exc:
        print(f"  Falló la descarga desde UCI: {exc}")

    if csv_bytes is None:
        print("Intentando con el espejo...")
        try:
            csv_bytes = _descargar(URL_ESPEJO)
        except Exception as exc:
            print(f"  El espejo también falló: {exc}")
            print(
                "\nDescárgalo a mano desde "
                "https://archive.ics.uci.edu/ml/datasets/bank+marketing\n"
                f"y guárdalo como {DESTINO}"
            )
            return 1

    DESTINO.write_bytes(csv_bytes)
    n_filas = csv_bytes.count(b"\n")
    print(f"Guardado en {DESTINO}  ({len(csv_bytes) / 1e6:.1f} MB, ~{n_filas:,} filas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
