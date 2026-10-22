from __future__ import annotations

import json
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

from .analyzer import Analyzer
from .calibration import load_calibration
from .engine import DEFAULT_PRESET, Detector
from .readers import Document, read_document
from .report import VIEWER, json_safe

MAX_UPLOAD = 50 * 1024 * 1024
ALLOWED = {".pdf", ".docx", ".txt", ".md"}


def create_app(preset: str = DEFAULT_PRESET, calibration: str | None = None) -> FastAPI:
    app = FastAPI(title="aidetect")
    jobs: dict[str, dict] = {}
    # One worker: the GPU runs one analysis at a time, others wait in the queue.
    pool = ThreadPoolExecutor(max_workers=1)
    state: dict = {}
    lock = threading.Lock()

    def analyzer() -> Analyzer:
        with lock:
            if "analyzer" not in state:
                det = Detector(preset)
                state["analyzer"] = Analyzer(det, load_calibration(det.preset, det.observer_name, det.performer_name, calibration))
            return state["analyzer"]

    def run(job_id: str, doc: Document) -> None:
        job = jobs[job_id]
        try:
            job["etat"] = "chargement"
            an = analyzer()
            job["etat"] = "analyse"

            def progress(done: int, total: int) -> None:
                job["fait"], job["total"] = done, total

            job["resultat"] = json_safe(an.analyze(doc, progress=progress))
            job["etat"] = "termine"
        except Exception as e:  # surfaced to the UI
            job["etat"], job["erreur"] = "erreur", str(e)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return VIEWER.read_text(encoding="utf-8")

    @app.post("/api/analyses")
    async def create(fichier: UploadFile | None = File(None), texte: str | None = Form(None)) -> dict:
        if fichier is not None and fichier.filename:
            suffix = Path(fichier.filename).suffix.lower()
            if suffix not in ALLOWED:
                raise HTTPException(400, f"Format non pris en charge : {suffix}")
            data = await fichier.read(MAX_UPLOAD + 1)
            if len(data) > MAX_UPLOAD:
                raise HTTPException(413, "Fichier trop volumineux (50 Mo max).")
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / f"document{suffix}"
                path.write_bytes(data)
                try:
                    doc = read_document(path)
                except Exception as e:
                    raise HTTPException(400, f"Lecture impossible : {e}")
            doc.name = Path(fichier.filename).name
        elif texte and texte.strip():
            doc = Document(texte.replace("\r\n", "\n").strip(), "Texte collé")
        else:
            raise HTTPException(400, "Fournissez un fichier ou un texte.")
        if not doc.text:
            raise HTTPException(400, "Aucun texte extractible (PDF scanné ?).")
        job_id = uuid.uuid4().hex
        jobs[job_id] = {"id": job_id, "etat": "en_attente", "fait": 0, "total": 0}
        pool.submit(run, job_id, doc)
        return jobs[job_id]

    @app.get("/api/analyses/{job_id}")
    def status(job_id: str) -> Response:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Analyse inconnue.")
        return Response(json.dumps(job, ensure_ascii=False), media_type="application/json")

    return app
