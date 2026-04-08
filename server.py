"""
server.py
---------
Flask REST API for the LegalRAG Property Due Diligence Platform
Runs on port 5000.

Routes:
  GET  /              → landing page (landing.html)
  GET  /landing       → landing page (alias)
  GET  /app           → dashboard SPA (index.html)
  GET  /app/<path>    → SPA assets
  GET  /api/...       → REST API
"""

import os, sys, json, uuid, time, threading
from datetime import datetime
from pathlib import Path

# ── .env loading ──────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
CASES_FILE    = BASE_DIR / "cases.json"
UPLOADS_DIR   = BASE_DIR / "uploads"
REPORTS_DIR   = BASE_DIR / "legal_reports"
FRONTEND_DIR  = BASE_DIR.parent / "frontend"
LANDING_FILE  = BASE_DIR.parent / "frontend" / "landing.html"

UPLOADS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── thread-safe JSON persistence ──────────────────────────────────────────────
_lock = threading.Lock()

def _read_cases():
    with _lock:
        if not CASES_FILE.exists():
            return []
        with open(CASES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

def _write_cases(cases):
    with _lock:
        with open(CASES_FILE, "w", encoding="utf-8") as f:
            json.dump(cases, f, indent=2, ensure_ascii=False)

def _get_case(case_id):
    for c in _read_cases():
        if c["id"] == case_id:
            return c
    return None

def _update_case(case_id, updates: dict):
    cases = _read_cases()
    for i, c in enumerate(cases):
        if c["id"] == case_id:
            cases[i].update(updates)
            _write_cases(cases)
            return cases[i]
    return None

# ── scoring helpers ───────────────────────────────────────────────────────────
def _score_to_level(lri_score: float) -> str:
    if lri_score >= 45:
        return "High"
    if lri_score >= 20:
        return "Medium"
    return "Low"

def _probability_to_10(probability: float) -> float:
    return round(1.0 + probability * 9.0, 1)

def _score_10_to_severity(score_10: float) -> str:
    if score_10 >= 7.0:
        return "High"
    if score_10 >= 4.0:
        return "Medium"
    return "Low"

# ── Gemini 429 retry wrapper ───────────────────────────────────────────────────
def _with_retry(fn, *args, retries=3, wait=12, **kwargs):
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                if attempt < retries:
                    print(f"[WARN] Rate limit hit (attempt {attempt}/{retries}). "
                          f"Waiting {wait}s before retry...")
                    time.sleep(wait)
                    wait = int(wait * 1.5)
                else:
                    raise
            else:
                raise

# ── background analysis thread ────────────────────────────────────────────────
def _run_analysis_thread(case_id: str, language: str = "English"):
    try:
        case = _get_case(case_id)
        if not case or not case.get("documents"):
            _update_case(case_id, {
                "analysisStatus": "Failed", "status": "Failed",
                "error": "No documents uploaded"
            })
            return

        raw_path = case["documents"][-1]["path"]
        rel_name = (raw_path
                    .replace("uploads\\", "")
                    .replace("uploads/", ""))
        doc_path = UPLOADS_DIR / rel_name
        if not doc_path.exists():
            alt = BASE_DIR / raw_path
            if alt.exists():
                doc_path = alt
            else:
                _update_case(case_id, {
                    "analysisStatus": "Failed", "status": "Failed",
                    "error": f"Document file not found: {raw_path}"
                })
                return

        _update_case(case_id, {"analysisStatus": "Analyzing", "status": "In Progress"})

        sys.path.insert(0, str(BASE_DIR))
        from legal_due_diligence_rag import LegalDueDiligenceSystem, _get_pdf_page_breaks
        from legal_risk_engine import run_risk_engine

        system = LegalDueDiligenceSystem(
            db_path=str(BASE_DIR / "vector_db"),
            collection_name="pdf_collection"
        )

        sale_deed_text    = system.extract_sale_deed_content(str(doc_path))
        rag_analysis      = _with_retry(system.analyze_sale_deed_with_rag, sale_deed_text)
        categorized_risks = _with_retry(system.classify_risks, rag_analysis, sale_deed_text)
        risk_summary      = _with_retry(system.generate_risk_summary, categorized_risks, sale_deed_text)

        real_page_breaks = _get_pdf_page_breaks(str(doc_path))
        lri_obj, lri_json_str = run_risk_engine(
            text=sale_deed_text,
            real_page_breaks=real_page_breaks,
            vectorizer=system.vectorizer,
            legal_db=system.legal_db,
            rag_queries_per_category=system.RISK_CATEGORY_RAG_QUERIES,
            categorized_risks=categorized_risks,
            print_cli=True,
            return_json=True,
        )

        key_findings = system.generate_key_findings(categorized_risks, lri_obj)

        category_scores = {}
        for cat_name, cat_obj in lri_obj.categories.items():
            score_10 = _probability_to_10(cat_obj.probability)
            severity = _score_10_to_severity(score_10)
            risk_entry = categorized_risks.get(cat_name, {})
            category_scores[cat_name] = {
                "score":         score_10,
                "severity":      severity,
                "analysis":      risk_entry.get("analysis", ""),
                "timestamp":     risk_entry.get("timestamp", datetime.now().isoformat()),
                "rag_ref_count": risk_entry.get("rag_ref_count", 0),
                "probability":          round(cat_obj.probability, 4),
                "threat_count":         cat_obj.threat_count,
                "gemini_severity_label": cat_obj.gemini_severity_label,
                "keyword_density":      round(cat_obj.keyword_density, 4),
                "rag_similarity_score": round(cat_obj.rag_similarity_score, 4),
            }

        risk_analysis_stored = {}
        for cat_name, entry in categorized_risks.items():
            risk_analysis_stored[cat_name] = {
                "analysis":        entry.get("analysis", ""),
                "timestamp":       entry.get("timestamp", ""),
                "rag_ref_count":   entry.get("rag_ref_count", 0),
            }

        lri_score  = lri_obj.lri_score
        risk_level = _score_to_level(lri_score)
        overall_10 = round(lri_score / 10.0, 1)

        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"legal_due_diligence_{case_id}_{ts}.pdf"
        report_path = REPORTS_DIR / report_name
        system.generate_pdf_report(
            sale_deed_path=str(doc_path),
            rag_analysis=rag_analysis,
            categorized_risks=categorized_risks,
            risk_summary=risk_summary,
            output_path=str(report_path),
            language=language,
            lri_obj=lri_obj,
            key_findings=key_findings,
        )

        report_rel = str(report_path.relative_to(BASE_DIR))
        lri_data   = json.loads(lri_json_str) if lri_json_str else {}

        _update_case(case_id, {
            "status":           "Completed",
            "analysisStatus":   "Completed",
            "riskLevel":        risk_level,
            "overallRiskScore": overall_10,
            "reportPath":       report_rel,
            "riskAnalysis":     risk_analysis_stored,
            "riskSummary":      risk_summary,
            "lriData":          lri_data,
            "keyFindings":      key_findings,
            "categoryScores":   category_scores,
            "completedAt":      datetime.now().isoformat(),
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        _update_case(case_id, {
            "analysisStatus": "Failed",
            "status":         "Failed",
            "error":          str(exc),
        })


# ════════════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════════════

# ── Landing page ──────────────────────────────────────────────────────────────
@app.route("/")
def root():
    """Redirect root to landing page."""
    return send_from_directory(str(FRONTEND_DIR), "landing.html")

@app.route("/landing")
def landing():
    """Landing / marketing page."""
    return send_from_directory(str(FRONTEND_DIR), "landing.html")

# ── Dashboard SPA ─────────────────────────────────────────────────────────────
@app.route("/app")
@app.route("/app/")
def app_index():
    """Serve the dashboard SPA."""
    return send_from_directory(str(FRONTEND_DIR), "index.html")

@app.route("/app/<path:path>")
def app_files(path):
    """Serve SPA assets; fall back to index.html for client-side routing."""
    full = FRONTEND_DIR / path
    if full.exists():
        return send_from_directory(str(FRONTEND_DIR), path)
    return send_from_directory(str(FRONTEND_DIR), "index.html")

# ── Static assets (JS, CSS, images) for both pages ───────────────────────────
@app.route("/<path:path>")
def static_files(path):
    full = FRONTEND_DIR / path
    if full.exists():
        return send_from_directory(str(FRONTEND_DIR), path)
    # Unknown paths → landing page
    return send_from_directory(str(FRONTEND_DIR), "landing.html")


# ── dashboard (extended) ──────────────────────────────────────────────────────
@app.route("/api/dashboard", methods=["GET"])
def dashboard():
    cases     = _read_cases()
    completed = [c for c in cases if c.get("analysisStatus") == "Completed"]
    high      = [c for c in completed if c.get("riskLevel") == "High"]
    medium    = [c for c in completed if c.get("riskLevel") == "Medium"]
    low       = [c for c in completed if c.get("riskLevel") == "Low"]
    pending   = [c for c in cases if c.get("analysisStatus") in ("Pending", "Analyzing")]
    scores    = [c.get("overallRiskScore", 0) for c in completed
                 if c.get("overallRiskScore") is not None]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0

    return jsonify({
        "totalCases":      len(cases),
        "completedCases":  len(completed),
        "highRisk":        len(high),
        "mediumRisk":      len(medium),
        "lowRisk":         len(low),
        "pendingReviews":  len(pending),
        "avgRiskScore":    avg_score,
        "highRiskPct":     round(len(high) / max(len(completed), 1) * 100, 1),
        "highRiskCases":   len(high),
        "mediumRiskCases": len(medium),
        "pendingCases":    len(pending),
    })


# ── /api/statistics ───────────────────────────────────────────────────────────
@app.route("/api/statistics", methods=["GET"])
def statistics():
    cases     = _read_cases()
    completed = [c for c in cases if c.get("analysisStatus") == "Completed"]
    high      = [c for c in completed if c.get("riskLevel") == "High"]
    medium    = [c for c in completed if c.get("riskLevel") == "Medium"]
    pending   = [c for c in cases if c.get("analysisStatus") in ("Pending", "Analyzing")]
    scores    = [c.get("overallRiskScore", 0) for c in completed
                 if c.get("overallRiskScore") is not None]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0

    return jsonify({
        "totalCases":      len(cases),
        "completedCases":  len(completed),
        "highRiskCases":   len(high),
        "mediumRiskCases": len(medium),
        "pendingCases":    len(pending),
        "avgRiskScore":    avg_score,
    })


# ── list / create cases ───────────────────────────────────────────────────────
@app.route("/api/cases", methods=["GET"])
def list_cases():
    cases  = _read_cases()
    cases.sort(key=lambda c: c.get("createdAt", ""), reverse=True)

    status = request.args.get("status")
    search = request.args.get("search", "").lower()
    if status and status != "all":
        cases = [c for c in cases if c.get("status") == status]
    if search:
        cases = [c for c in cases
                 if search in c.get("caseName", "").lower()
                 or search in c.get("propertyAddress", "").lower()]

    return jsonify(cases)


@app.route("/api/cases", methods=["POST"])
def create_case():
    body = request.get_json(force=True, silent=True) or {}
    case = {
        "id":               str(uuid.uuid4()),
        "caseName":         body.get("caseName", "Unnamed Case"),
        "propertyAddress":  body.get("propertyAddress", ""),
        "status":           "In Progress",
        "createdAt":        datetime.now().isoformat(),
        "riskLevel":        "Low",
        "overallRiskScore": 0,
        "documents":        [],
        "reportPath":       None,
        "analysisStatus":   "Pending",
        "riskAnalysis":     None,
        "lriData":          None,
        "keyFindings":      None,
        "categoryScores":   None,
        "riskSummary":      None,
    }
    cases = _read_cases()
    cases.append(case)
    _write_cases(cases)
    return jsonify(case), 201


# ── single case ───────────────────────────────────────────────────────────────
@app.route("/api/cases/<case_id>", methods=["GET"])
def get_case(case_id):
    case = _get_case(case_id)
    if not case:
        return jsonify({"error": "Case not found"}), 404
    return jsonify(case)


@app.route("/api/cases/<case_id>", methods=["DELETE"])
def delete_case(case_id):
    cases     = _read_cases()
    new_cases = [c for c in cases if c["id"] != case_id]
    if len(new_cases) == len(cases):
        return jsonify({"error": "Not found"}), 404
    _write_cases(new_cases)
    return jsonify({"message": "Deleted"})


# ── upload ────────────────────────────────────────────────────────────────────
@app.route("/api/cases/<case_id>/upload", methods=["POST"])
def upload_document(case_id):
    case = _get_case(case_id)
    if not case:
        return jsonify({"error": "Case not found"}), 404

    files = request.files.getlist("file") or request.files.getlist("files")
    if not files:
        return jsonify({"error": "No file provided"}), 400

    existing_docs = case.get("documents", [])
    new_docs = []
    for f in files:
        new_name = f"{case_id}_{f.filename}"
        dest     = UPLOADS_DIR / new_name
        f.save(str(dest))
        new_docs.append({
            "filename":   f.filename,
            "path":       f"uploads/{new_name}",
            "uploadedAt": datetime.now().isoformat(),
        })

    _update_case(case_id, {
        "documents":      existing_docs + new_docs,
        "analysisStatus": "Pending",
    })
    return jsonify({"message": f"{len(new_docs)} file(s) uploaded",
                    "documents": existing_docs + new_docs})


# ── trigger analysis (async) ──────────────────────────────────────────────────
@app.route("/api/cases/<case_id>/analyze", methods=["POST"])
def analyze_case(case_id):
    case = _get_case(case_id)
    if not case:
        return jsonify({"error": "Case not found"}), 404

    if case.get("analysisStatus") == "Analyzing":
        return jsonify({"message": "Analysis already in progress"}), 200

    body     = request.get_json(force=True, silent=True) or {}
    language = body.get("language", "English")

    t = threading.Thread(
        target=_run_analysis_thread,
        args=(case_id, language),
        daemon=True
    )
    t.start()
    return jsonify({"message": "Analysis started", "caseId": case_id}), 202


# ── poll status ───────────────────────────────────────────────────────────────
@app.route("/api/cases/<case_id>/status", methods=["GET"])
def case_status(case_id):
    case = _get_case(case_id)
    if not case:
        return jsonify({"error": "Case not found"}), 404
    return jsonify({
        "caseId":           case_id,
        "analysisStatus":   case.get("analysisStatus"),
        "status":           case.get("status"),
        "overallRiskScore": case.get("overallRiskScore"),
        "riskLevel":        case.get("riskLevel"),
        "error":            case.get("error"),
    })


# ── full report ───────────────────────────────────────────────────────────────
@app.route("/api/cases/<case_id>/report", methods=["GET"])
def case_report(case_id):
    case = _get_case(case_id)
    if not case:
        return jsonify({"error": "Case not found"}), 404
    return jsonify(case)


# ── LRI data ──────────────────────────────────────────────────────────────────
@app.route("/api/cases/<case_id>/lri", methods=["GET"])
def get_lri(case_id):
    case = _get_case(case_id)
    if not case:
        return jsonify({"error": "Case not found"}), 404
    lri_data = case.get("lriData")
    if not lri_data:
        return jsonify({"error": "LRI data not yet available. Run analysis first."}), 404
    return jsonify(lri_data)


# ── download report ───────────────────────────────────────────────────────────
@app.route("/api/cases/<case_id>/download", methods=["GET"])
def download_report(case_id):
    case = _get_case(case_id)
    if not case or not case.get("reportPath"):
        return jsonify({"error": "Report not available"}), 404
    report = BASE_DIR / case["reportPath"]
    if not report.exists():
        return jsonify({"error": "PDF file missing on disk"}), 404
    return send_file(
        str(report),
        as_attachment=True,
        download_name=f"legal_report_{case_id}.pdf",
        mimetype="application/pdf",
    )


@app.route("/api/cases/<case_id>/download-pdf", methods=["GET"])
def download_pdf(case_id):
    return download_report(case_id)


# ── serve original uploaded document ─────────────────────────────────────────
@app.route("/api/cases/<case_id>/document", methods=["GET"])
def serve_document(case_id):
    case = _get_case(case_id)
    if not case or not case.get("documents"):
        return jsonify({"error": "No document"}), 404
    raw_path = case["documents"][-1]["path"]
    rel_name = raw_path.replace("uploads/", "").replace("uploads\\", "")
    doc_path = UPLOADS_DIR / rel_name
    if not doc_path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(str(doc_path), mimetype="application/pdf")


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  LegalRAG Server")
    print()
    print("  Landing Page  →  http://localhost:5000/")
    print("  Dashboard     →  http://localhost:5000/app")
    print("  API           →  http://localhost:5000/api/...")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)