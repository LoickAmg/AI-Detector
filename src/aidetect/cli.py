from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .engine import DEFAULT_PRESET, PRESETS


def _pct(p) -> str:
    return "  —  " if p is None or p != p else f"{p * 100:4.0f} %"


def _progress(done: int, total: int) -> None:
    print(f"\r  séquence {done}/{total}", end="" if done < total else "\n", file=sys.stderr, flush=True)


def cmd_analyser(args) -> None:
    from .analyzer import Analyzer
    from .engine import Detector
    from .readers import Document, read_document
    from .report import render_html, to_json

    if args.texte:
        doc = Document(args.texte, "Texte")
    elif args.fichier == "-":
        doc = Document(sys.stdin.read().strip(), "stdin")
    elif args.fichier:
        doc = read_document(args.fichier)
    else:
        sys.exit("Indiquez un fichier, « - » pour stdin, ou --texte.")
    if not doc.text:
        sys.exit("Aucun texte extractible (PDF scanné ?).")

    print(f"Chargement des modèles ({args.preset})…", file=sys.stderr)
    analyzer = Analyzer(Detector(args.preset))
    r = analyzer.analyze(doc, sequence_tokens=args.sequence, progress=_progress)

    g, s = r["global"], r["synthese"]
    print(f"\n{r['document']} — {r['tokens']} tokens" + (f", {r['pages']} pages" if r["pages"] else ""))
    ic = g.get("intervalle_90")
    ic_txt = f", IC 90 % {ic[0] * 100:.0f}–{ic[1] * 100:.0f} %" if ic else ""
    print(f"Verdict global : {g['verdict']}  ({_pct(g['probabilite_ia']).strip()}{ic_txt}, fiabilité {g['fiabilite']})")
    if r["caracteres_suspects"]:
        print(f"Caractères inhabituels neutralisés : {r['caracteres_suspects']}")
    print(f"Part estimée IA : {_pct(s['part_ia_estimee']).strip()} — {s['sequences_ia']}/{s['sequences']} séquences IA")
    if not r["calibre"]:
        print("Attention : seuils non calibrés pour ces modèles (aidetect calibrer).")
    if s["sequences"] > 1:
        print("\n  #   page  tokens  proba   verdict")
        for q in r["sequences"]:
            page = str(q["page"]) if q["page"] else "—"
            print(f"{q['index']:>3}  {page:>5}  {q['tokens']:>6}  {_pct(q['probabilite_ia'])}  {q['verdict']}")

    if args.rapport:
        Path(args.rapport).write_text(render_html(r), encoding="utf-8")
        print(f"\nRapport : {Path(args.rapport).resolve()}")
    if args.json:
        Path(args.json).write_text(to_json(r), encoding="utf-8")


def cmd_serveur(args) -> None:
    import uvicorn

    from .server import create_app

    print(f"Interface : http://127.0.0.1:{args.port}", file=sys.stderr)
    uvicorn.run(create_app(args.preset), host="127.0.0.1", port=args.port)


def _load_labeled(args) -> list[tuple[str, int, str, str]]:
    from .readers import read_document

    items = []
    if args.jsonl:
        for i, line in enumerate(Path(args.jsonl).read_text(encoding="utf-8").splitlines()):
            if line.strip():
                row = json.loads(line)
                items.append((row["text"], int(row["label"]), str(row.get("group", f"l{i}")), row.get("source", "jsonl")))
    for folder, label in ((args.humain, 0), (args.ia, 1)):
        if folder:
            for path in sorted(Path(folder).iterdir()):
                if path.suffix.lower() in {".txt", ".md", ".pdf", ".docx"}:
                    items.append((read_document(path).text, label, f"{label}:{path.name}", Path(folder).name))
    return items


def cmd_calibrer(args) -> None:
    import random

    from .calibration import DEFAULT_VARIABLES, calibration_path, fit_calibration
    from .engine import Detector

    if args.depuis_scores:
        rows = [json.loads(l) for l in Path(args.depuis_scores).read_text(encoding="utf-8").splitlines() if l.strip()]
        detector = None
    else:
        items = _load_labeled(args)
        if len({it[1] for it in items}) < 2:
            sys.exit("Il faut des textes humains (label 0) ET IA (label 1).")
        detector = Detector(args.preset)
        rng = random.Random(0)
        rows = []
        for k, (text, label, group, source) in enumerate(items):
            print(f"\r  texte {k + 1}/{len(items)}", end="", file=sys.stderr, flush=True)
            cum = detector.token_stats(text).cumulative()
            n = len(cum.count) - 1
            meta = {"label": label, "group": group, "source": source}
            # Windows of several lengths so the calibration learns how confidence depends on length.
            for length in (40, 80, 150, 300, 600):
                if n < length + 1:
                    continue
                for _ in range(min(3, n // length)):
                    a = rng.randint(1, n - length)
                    rows.append({**cum.span(a, a + length), **meta})
            if n >= 20:
                rows.append({**cum.span(1, n), **meta, "complet": True})
        print(file=sys.stderr)
        if args.scores:
            Path(args.scores).parent.mkdir(parents=True, exist_ok=True)
            Path(args.scores).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    observer, performer = PRESETS[args.preset]
    if detector is not None:
        observer, performer = detector.observer_name, detector.performer_name
    variables = args.variables or DEFAULT_VARIABLES.get(args.preset, "bino+fdg")
    calib = fit_calibration(rows, observer, performer, variables=variables)
    out = Path(args.sortie) if args.sortie else calibration_path(args.preset)
    calib.save(out)
    print(json.dumps(calib.metrics, indent=2, ensure_ascii=False))
    print(f"Calibration enregistrée : {out}")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="aidetect", description="Détection locale de texte généré par IA.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("analyser", help="Analyser un texte ou un document (pdf, docx, txt, md)")
    p.add_argument("fichier", nargs="?")
    p.add_argument("--texte")
    p.add_argument("--rapport", help="Chemin du rapport HTML")
    p.add_argument("--json", help="Chemin de l'export JSON")
    p.add_argument("--sequence", type=int, default=300, help="Taille cible d'une séquence en tokens")
    p.add_argument("--preset", choices=PRESETS, default=DEFAULT_PRESET)
    p.set_defaults(func=cmd_analyser)

    p = sub.add_parser("serveur", help="Lancer l'interface web locale")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--preset", choices=PRESETS, default=DEFAULT_PRESET)
    p.set_defaults(func=cmd_serveur)

    p = sub.add_parser("calibrer", help="Ajuster les probabilités sur des textes étiquetés")
    p.add_argument("--humain", help="Dossier de textes écrits par des humains")
    p.add_argument("--ia", help="Dossier de textes générés par IA")
    p.add_argument("--jsonl", help='Fichier JSONL : {"text": ..., "label": 0|1, "group": ...}')
    p.add_argument("--scores", help="Enregistrer les scores bruts (JSONL) pour réutilisation")
    p.add_argument("--depuis-scores", help="Recalibrer depuis des scores bruts déjà calculés")
    p.add_argument("--variables", choices=["bino+fdg", "bino"], help="Variables du modèle de calibration (défaut selon le preset)")
    p.add_argument("--sortie")
    p.add_argument("--preset", choices=PRESETS, default=DEFAULT_PRESET)
    p.set_defaults(func=cmd_calibrer)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
