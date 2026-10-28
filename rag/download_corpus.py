"""
Descarga el corpus de la fase RAG desde la API REST de CIMA (AEMPS).

Para cada principio activo de la lista DRUGS:
    1. Busca en CIMA un medicamento comercializado, autorizado y de uso parenteral.
    2. Descarga la ficha técnica segmentada (docSegmentado/contenido/1).
    3. Guarda el JSON crudo en  rag/raw/<nregistro>.json          (respaldo, fuera de corpus/)
    4. Guarda las secciones especificadas en SECTIONS limpias en  rag/corpus/<slug>_<nregistro>.md
    5. Escribe rag/corpus_manifest.csv con exactamente qué medicamentos se han usado.

Reproducibilidad: CIMA cambia con el tiempo. Una vez generado el manifiesto, puedes
volver a descargar los MISMOS medicamentos con:   python download_corpus.py --from-manifest

Uso:
    uv run python -m rag.download_corpus                  # búsqueda + descarga
    uv run python -m rag.download_corpus --from-manifest  # re-descarga usando el manifiesto
    uv run python -m rag.download_corpus --limit 5        # prueba rápida con 5 principios activos

Documentación API: https://cima.aemps.es/cima/resources/docs/CIMA_REST_API.pdf
"""
import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import requests

BASE = "https://cima.aemps.es/cima/rest"
ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "corpus"
RAW_DIR = ROOT / "raw"
MANIFEST = ROOT / "corpus_manifest.csv"

SECTIONS = ["2", "4.2", "6.2", "6.3", "6.4", "6.5", "6.6"]
PAUSE = 0.4

# Vías que aceptamos como "parenterales" (se compara contra el nombre de la vía en CIMA)
VIAS_OK = ("INTRAVEN", "INTRAMUSC", "SUBCUTAN", "PARENTERAL")

# (principio activo para practiv1, nº de principios activos, texto que debe contener el nombre)
DRUGS = [
    # --- Antibióticos y antiinfecciosos IV ---
    ("ceftriaxona", 1, None),
    ("cefazolina", 1, None),
    ("cefotaxima", 1, None),
    ("ceftazidima", 1, None),
    ("cefuroxima", 1, None),
    ("amoxicilina", 2, "clavul"),
    ("piperacilina", 2, "tazobactam"),
    ("meropenem", 1, None),
    ("imipenem", 2, "cilastatina"),
    ("ertapenem", 1, None),
    ("vancomicina", 1, None),
    ("gentamicina", 1, None),
    ("amikacina", 1, None),
    ("tobramicina", 1, None),
    ("levofloxacino", 1, None),
    ("ciprofloxacino", 1, None),
    ("metronidazol", 1, None),
    ("linezolid", 1, None),
    ("clindamicina", 1, None),
    ("azitromicina", 1, None),
    ("fluconazol", 1, None),
    ("aciclovir", 1, None),
    ("daptomicina", 1, None),
    # --- Analgésicos y antiinflamatorios IV ---
    ("paracetamol", 1, None),
    ("metamizol", 1, None),
    ("ibuprofeno", 1, None),
    ("dexketoprofeno", 1, None),
    ("tramadol", 1, None),
    ("morfina", 1, None),
    ("fentanilo", 1, None),
    # --- Vasopresores / cardiovasculares ---
    ("noradrenalina", 1, None),
    ("adrenalina", 1, None),
    ("dopamina", 1, None),
    ("dobutamina", 1, None),
    ("nitroglicerina", 1, None),
    ("amiodarona", 1, None),
    ("esmolol", 1, None),
    ("furosemida", 1, None),
    # --- Electrolitos ---
    ("cloruro de potasio", 1, None),
    ("magnesio sulfato", 1, None),
    ("sodio bicarbonato", 1, None),
    # --- Anticoagulantes ---
    ("heparina sodica", 1, None),
    ("enoxaparina", 1, None),
    # --- Sedantes / anestésicos ---
    ("midazolam", 1, None),
    ("propofol", 1, None),
    ("dexmedetomidina", 1, None),
    ("ketamina", 1, None),
]

session = requests.Session()
session.headers.update({
    "Accept": "application/json",
    "User-Agent": "practica-rag-enfermeria/1.0 (proyecto academico)",
})


def get_json(path, params=None, retries=4):
    """GET con reintentos y pausa. Devuelve el JSON o None si no hay contenido."""
    url = f"{BASE}/{path}"
    last = None
    for i in range(retries):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 204:
                return None
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
            else:
                r.raise_for_status()
        except (requests.RequestException, ValueError) as e:
            last = e
        time.sleep(PAUSE)
    raise RuntimeError(f"Fallo al pedir {url} {params}: {last}")


class _HTMLToText(HTMLParser):
    BLOCK = {"p", "div", "br", "tr", "table", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag == "li":
            self.out.append("\n- ")
        elif tag in ("td", "th"):
            self.out.append(" | ")
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.out.append(data)


def html_to_text(html):
    p = _HTMLToText()
    p.feed(html or "")
    text = "".join(p.out)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t\u00a0]+", " ", line).strip()
        if " | " in line or line.startswith("|"):
            line = line.strip(" |")  # filas de tabla: "a | b | c"
        lines.append(line)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def slugify(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60]


def fecha_ficha(med):
    """Fecha de última modificación de la ficha técnica según CIMA (epoch -> ISO)."""
    for d in med.get("docs") or []:
        if d.get("tipo") == 1 and d.get("fecha"):
            t = d["fecha"]
            t = t / 1000 if t > 1e11 else t  # CIMA suele dar milisegundos
            return datetime.fromtimestamp(t, timezone.utc).date().isoformat()
    return ""


def norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def es_candidato(m, must):
    if must and must not in norm(m.get("nombre")):
        return False
    if not any(d.get("tipo") == 1 and d.get("secc") for d in m.get("docs") or []):
        return False
    vias = " ".join((v.get("nombre") or "").upper() for v in m.get("viasAdministracion") or [])
    return any(v in vias for v in VIAS_OK)


def es_iv(m):
    vias = " ".join((v.get("nombre") or "").upper() for v in m.get("viasAdministracion") or [])
    return "INTRAVEN" in vias


def buscar_medicamento(practiv, npactiv, must):
    cands = []
    for pagina in range(1, 4):
        data = get_json("medicamentos", {
            "practiv1": practiv, "npactiv": npactiv,
            "comerc": 1, "autorizados": 1, "pagina": pagina,
        })
        if not data or not data.get("resultados"):
            break
        cands += [m for m in data["resultados"] if es_candidato(m, must)]
        if pagina * data.get("tamanioPagina", 25) >= data.get("totalFilas", 0):
            break
    if not cands:
        return None

    cands.sort(key=lambda m: (not es_iv(m), m["nregistro"]))
    return cands[0]


def detalle_medicamento(nregistro):
    return get_json("medicamento", {"nregistro": nregistro})


def descargar_ficha(nregistro):
    data = get_json("docSegmentado/contenido/1", {"nregistro": nregistro})
    if data is None:
        return None
    if isinstance(data, dict):
        data = data.get("secciones") or [data]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / f"{nregistro}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data


def seccion_deseada(num):
    return any(num == w or num.startswith(w + ".") for w in SECTIONS)


def escribir_markdown(med, practiv, secciones):
    nreg = med["nregistro"]
    forma = (med.get("formaFarmaceutica") or {}).get("nombre", "")
    vias = ", ".join(v.get("nombre", "") for v in med.get("viasAdministracion") or [])
    url = next((d.get("urlHtml") or d.get("url") for d in med.get("docs") or []
                if d.get("tipo") == 1), "")
    partes = [
        f"# {med['nombre']}",
        "",
        f"- Principio activo (búsqueda): {practiv}",
        f"- Nº de registro CIMA: {nreg}",
        f"- Laboratorio titular: {med.get('labtitular', '')}",
        f"- Forma farmacéutica: {forma}",
        f"- Vías de administración: {vias}",
        f"- Fuente: CIMA (AEMPS), ficha técnica. {url}",
        f"- Última actualización de la ficha (según CIMA): {fecha_ficha(med) or 'no disponible'}",
        f"- Descargado: {date.today().isoformat()}",
        "",
    ]
    guardadas = []
    for s in sorted(secciones, key=lambda s: s.get("orden", 0)):
        num = str(s.get("seccion", ""))
        if not seccion_deseada(num):
            continue

        texto = html_to_text(s.get("contenido", ""))
        nivel = "##" if num.count(".") <= 1 else "###"
        partes += [f"{nivel} {num} {s.get('titulo', '').strip()}", "",
                   texto or "(sin contenido)", ""]

        guardadas.append(num)

    if not guardadas:
        return None, []
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    fichero = CORPUS_DIR / f"{slugify(practiv)}_{nreg}.md"
    fichero.write_text("\n".join(partes), encoding="utf-8")
    return fichero, guardadas


def leer_manifiesto():
    with MANIFEST.open(encoding="utf-8") as f:
        return [(r["principio_activo"], r["nregistro"]) for r in csv.DictReader(f)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--from-manifest", action="store_true",
                    help="Re-descarga los medicamentos listados en corpus_manifest.csv")
    ap.add_argument("--limit", type=int, help="Procesa solo los N primeros (para probar)")
    args = ap.parse_args()

    if args.from_manifest:
        trabajo = [(p, None, None, n) for p, n in leer_manifiesto()]
    else:
        trabajo = [(p, np_, must, None) for p, np_, must in DRUGS]
    if args.limit:
        trabajo = trabajo[: args.limit]

    filas, fallos = [], []
    for i, (practiv, npactiv, must, nreg_fijo) in enumerate(trabajo, 1):
        print(f"[{i}/{len(trabajo)}] {practiv} ...", end=" ", flush=True)
        try:
            if nreg_fijo:
                med = detalle_medicamento(nreg_fijo)
            else:
                med = buscar_medicamento(practiv, npactiv, must)
                if med:
                    med = detalle_medicamento(med["nregistro"]) or med
            if not med:
                raise LookupError("sin candidato válido en CIMA")
            secciones = descargar_ficha(med["nregistro"])
            if not secciones:
                raise LookupError("sin ficha técnica segmentada")
            fichero, guardadas = escribir_markdown(med, practiv, secciones)
            if not fichero:
                raise LookupError("ninguna de las secciones pedidas existe")
        except Exception as e:  # seguimos con el resto, informamos al final
            print(f"FALLO ({e})")
            fallos.append((practiv, str(e)))
            continue
        faltan = [s for s in SECTIONS if not any(g == s or g.startswith(s + ".") for g in guardadas)]
        print(f"OK  {med['nombre'][:50]} [{med['nregistro']}]"
              + (f"  (faltan secciones: {', '.join(faltan)})" if faltan else ""))
        filas.append({
            "principio_activo": practiv,
            "nregistro": med["nregistro"],
            "nombre": med["nombre"],
            "laboratorio": med.get("labtitular", ""),
            "forma_farmaceutica": (med.get("formaFarmaceutica") or {}).get("nombre", ""),
            "vias": "; ".join(v.get("nombre", "") for v in med.get("viasAdministracion") or []),
            "secciones_guardadas": " ".join(guardadas),
            "fichero": fichero.name,
            "fecha_ficha": fecha_ficha(med),
            "fecha_descarga": date.today().isoformat(),
        })

    if filas and not args.limit:
        with MANIFEST.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
            w.writeheader()
            w.writerows(filas)
        print(f"\nManifiesto escrito en {MANIFEST}")

    print(f"\nDescargados: {len(filas)} / {len(trabajo)}")
    if fallos:
        print("Sin descargar (revisa el nombre del principio activo o quítalo de la lista):")
        for p, e in fallos:
            print(f"  - {p}: {e}")
    return 0 if filas else 1


if __name__ == "__main__":
    sys.exit(main())