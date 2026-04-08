# legal_due_diligence_rag.py
"""
RAG-Enhanced Legal Due Diligence System for Property Sale Deed Analysis
With working Tamil/multilingual translation using deep-translator
"""

import os
import time
from datetime import datetime
from pathlib import Path
from google import genai as _genai

from typing import List, Dict
import json
from deep_translator import GoogleTranslator

# PDF Generation Libraries
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Import your existing vectorizer
from vb import PDFVectorizer

# ── NEW: algorithmic risk engine (Feature 1 + Feature 2) ──────────────────────
from legal_risk_engine import run_risk_engine, lri_to_json, LegalRiskIndex


def _init_gemini():
    """Initializes Gemini client using the new google-genai SDK"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    return _genai.Client(api_key=api_key)


def _get_pdf_page_breaks(file_path: str) -> list:
    """
    Returns character-offset positions where each PDF page begins.
    Falls back to [0] for non-PDF files or if fitz is unavailable.
    Used by the LRI engine to map threats to real page numbers.
    """
    if Path(file_path).suffix.lower() != ".pdf":
        return [0]
    try:
        import fitz
        doc = fitz.open(file_path)
        breaks, cumulative = [0], 0
        for page in doc:
            cumulative += len(page.get_text())
            breaks.append(cumulative)
        doc.close()
        return breaks
    except Exception:
        return [0]


class LegalDueDiligenceSystem:
    """Complete legal due diligence system with RAG and risk assessment"""

    RISK_CATEGORIES = {
        "Ownership Disputes": [
            "unclear title chain", "multiple claimants", "succession issues",
            "partition disputes", "joint ownership conflicts"
        ],
        "Litigation History": [
            "pending cases", "court orders", "injunctions",
            "stay orders", "criminal proceedings"
        ],
        "Construction Compliance": [
            "building plan approval", "occupancy certificate", "completion certificate",
            "unauthorized construction", "setback violations", "FSI violations"
        ],
        "Tax Obligations": [
            "property tax arrears", "stamp duty", "registration charges",
            "capital gains tax", "TDS obligations"
        ],
        "Government Notifications": [
            "acquisition notice", "demolition order", "land use change",
            "town planning schemes", "road widening", "reservation for public purpose"
        ]
    }

    RISK_CATEGORY_RAG_QUERIES = {
        "Ownership Disputes": [
            "title deed ownership chain verification",
            "property succession inheritance law India",
            "joint ownership partition dispute resolution",
        ],
        "Litigation History": [
            "property litigation pending court cases",
            "injunction stay order property transaction",
            "encumbrance certificate legal disputes",
        ],
        "Construction Compliance": [
            "building plan approval RERA compliance",
            "occupancy certificate completion certificate requirements",
            "unauthorized construction FSI setback violations penalties",
        ],
        "Tax Obligations": [
            "property tax stamp duty registration charges",
            "capital gains tax TDS property sale",
            "property tax arrears liability buyer seller",
        ],
        "Government Notifications": [
            "government land acquisition notice property",
            "town planning scheme road widening reservation",
            "demolition order land use change notification",
        ],
    }

    LANGUAGES = {
        "English":              {"code": "en", "font": None},
        "हिन्दी (Hindi)":       {"code": "hi", "font": "NotoSansDevanagari"},
        "தமிழ் (Tamil)":        {"code": "ta", "font": "NotoSansTamil"},
        "বাংলা (Bengali)":      {"code": "bn", "font": "NotoSansBengali"},
        "తెలుగు (Telugu)":      {"code": "te", "font": "NotoSansTelugu"},
        "मराठी (Marathi)":      {"code": "mr", "font": "NotoSansDevanagari"},
        "ગુજરાતી (Gujarati)":   {"code": "gu", "font": "NotoSansGujarati"}
    }

    _GENERATION_CONFIG = {
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 8192,
    }

    def __init__(
        self,
        db_path: str = "./vector_db",
        collection_name: str = "pdf_collection"
    ):
        """Initialize the system"""
        self.db_path = db_path
        self.collection_name = collection_name

        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self._genai_client = _genai.Client(api_key=gemini_api_key)
        print("[OK] Initialized Gemini client (google-genai SDK)")

        print("[INFO] Checking available Gemini models...")
        try:
            available_models = [m.name for m in self._genai_client.models.list()]
            print(f"[OK] Found {len(available_models)} models")

            preferred_models = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-001']
            self._model_name = None
            for model_name in preferred_models:
                if any(model_name in m for m in available_models):
                    self._model_name = model_name
                    break

            if not self._model_name:
                self._model_name = available_models[0].replace('models/', '')
        except Exception as e:
            print(f"[WARN] Could not list models: {e}")
            self._model_name = 'gemini-2.0-flash'

        print(f"[OK] Initialized {self._model_name} for text generation")

        self.vectorizer = PDFVectorizer(db_path=db_path, collection_name=collection_name)

        try:
            self.legal_db = self.vectorizer.load_existing_database()
            print(f"[OK] Loaded legal knowledge base with {self.legal_db.count()} documents")
        except Exception as e:
            print(f"[WARN] Could not load legal database: {e}")
            self.legal_db = None

    # ─────────────────────────────────────────────────────────────────────────
    # CORE: generate_content  ── UNCHANGED
    # ─────────────────────────────────────────────────────────────────────────
    def generate_content(self, prompt: str):
        """
        Calls the new google-genai SDK and returns an object with a .text attribute.
        """
        response = self._genai_client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=self._GENERATION_CONFIG,
        )

        class _Response:
            def __init__(self, text):
                self.text = text

        return _Response(response.text)

    # ─────────────────────────────────────────────────────────────────────────
    # CORE: query_legal_database  ── UNCHANGED
    # ─────────────────────────────────────────────────────────────────────────
    def query_legal_database(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Query the RAG vector database for relevant legal provisions.
        Returns a list of dicts with 'content', 'source', 'relevance'.
        Returns [] if no database is loaded (graceful degradation).
        """
        if not self.legal_db:
            return []

        try:
            results = self.vectorizer.query_database(self.legal_db, query, n_results=n_results)
            legal_docs = []
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                legal_docs.append({
                    "content": doc,
                    "source": metadata.get('source', 'Unknown'),
                    "relevance": "High"
                })
            return legal_docs
        except Exception as e:
            print(f"    [WARN] RAG query failed for '{query}': {e}")
            return []

    def _fetch_rag_for_category(self, category: str) -> List[Dict]:
        """
        Runs all RAG queries for a given risk category and de-duplicates results.
        Returns a merged list of unique legal provisions relevant to that category.
        """
        queries = self.RISK_CATEGORY_RAG_QUERIES.get(category, [])
        seen_content = set()
        merged_docs = []

        for query in queries:
            docs = self.query_legal_database(query, n_results=3)
            for doc in docs:
                key = doc['content'][:120].strip()
                if key not in seen_content:
                    seen_content.add(key)
                    merged_docs.append(doc)

        return merged_docs

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: RAG analysis  ── UNCHANGED
    # ─────────────────────────────────────────────────────────────────────────
    def analyze_sale_deed_with_rag(self, sale_deed_text: str) -> Dict:
        """
        Analyzes the sale deed across 5 legal aspects.
        Each aspect fetches targeted provisions from the RAG DB before prompting.
        """
        print("\n[INFO] Analyzing sale deed with RAG system...")

        analysis_aspects = [
            "title verification and ownership chain",
            "encumbrance and liens",
            "property tax compliance",
            "building approvals and construction compliance",
            "litigation and legal disputes"
        ]

        rag_analysis = {}

        for aspect in analysis_aspects:
            print(f"  - Checking: {aspect}")
            legal_refs = self.query_legal_database(aspect, n_results=3)

            prompt = f"""
            You are an expert legal analyst. Write a clear, professional analysis in paragraph form.

            Analyze this sale deed for: {aspect}

            SALE DEED CONTENT:
            {sale_deed_text[:4000]}

            RELEVANT LEGAL PROVISIONS FROM KNOWLEDGE BASE:
            {json.dumps([ref['content'][:700] for ref in legal_refs], indent=2)}

            Write 3-5 clear paragraphs covering:

            First paragraph: Introduce what you examined and the overall findings.

            Second paragraph: Discuss the compliance status with relevant laws and regulations.
            Explain what the document contains and what might be missing.

            Third paragraph: Identify any potential risks or concerns. Be specific about
            what issues were found and their implications.

            Fourth paragraph (if applicable): Reference the specific laws, acts, or regulations
            that apply to this aspect. Explain how they relate to the findings.

            Final paragraph: Provide clear, actionable recommendations. Be specific about
            what steps should be taken and why.

            IMPORTANT: Write in flowing paragraphs, NOT bullet points or lists.
            Use professional but accessible language. Each paragraph should be 3-5 sentences.
            """

            try:
                response = self.generate_content(prompt)
                formatted_text = self.reformat_for_readability(response.text)

                rag_analysis[aspect] = {
                    "analysis": formatted_text,
                    "legal_references": legal_refs
                }
                time.sleep(1)
            except Exception as e:
                print(f"    [WARN] Error: {e}")
                rag_analysis[aspect] = {
                    "analysis": f"Analysis unavailable: {str(e)}",
                    "legal_references": legal_refs
                }

        return rag_analysis

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Risk classification  ── UNCHANGED
    # ─────────────────────────────────────────────────────────────────────────
    def classify_risks(self, rag_analysis: Dict, sale_deed_text: str) -> Dict:
        print("\n[INFO] Classifying risks with RAG-grounded legal provisions...")

        all_analysis = "\n\n".join([
            f"=== {aspect.upper()} ===\n{data['analysis']}"
            for aspect, data in rag_analysis.items()
        ])

        categorized_risks = {}

        for category, keywords in self.RISK_CATEGORIES.items():
            print(f"  - Analyzing: {category}")

            category_legal_refs = self._fetch_rag_for_category(category)
            ref_count = len(category_legal_refs)
            print(f"    [RAG] Retrieved {ref_count} legal provision(s) for '{category}'")

            if category_legal_refs:
                formatted_refs = "\n\n".join([
                    f"[Source: {ref['source']}]\n{ref['content'][:600]}"
                    for ref in category_legal_refs
                ])
            else:
                formatted_refs = "No specific provisions retrieved from knowledge base."

            prompt = f"""
            You are a senior legal risk analyst with expertise in Indian property law.
            Your analysis must be grounded in the legal provisions provided below.

            ── RISK CATEGORY ────────────────────────────────────────────────────────
            Category: {category}
            Focus areas: {', '.join(keywords)}

            ── RETRIEVED LEGAL PROVISIONS (from RAG knowledge base) ─────────────────
            These are the most relevant laws, acts, and legal provisions retrieved
            specifically for this risk category. Base your risk assessment on these.

            {formatted_refs}

            ── PRIOR ANALYSIS (from general sale deed review) ────────────────────────
            {all_analysis}

            ── SALE DEED EXCERPT ─────────────────────────────────────────────────────
            {sale_deed_text[:2500]}

            ── INSTRUCTIONS ──────────────────────────────────────────────────────────
            Write a 3-4 paragraph risk assessment:

            First paragraph: State the overall severity level (High/Medium/Low/None).
            Reference the specific legal provisions retrieved above that are most
            relevant to this category. Explain how they apply to this transaction.

            Second paragraph: Describe the specific issues found. Cite relevant sections
            of the legal provisions where applicable. Be precise about what was found
            in the sale deed and what is missing or non-compliant.

            Third paragraph: Explain the legal consequences if these issues are not
            addressed. Reference specific acts, penalties, or legal outcomes that
            could follow based on the retrieved provisions.

            Final paragraph: Provide actionable recommendations with explicit reference
            to the applicable law. State what must be done, the legal basis for it,
            and the timeline (immediate / 30 days / before registration).

            Write in flowing professional paragraphs. No bullet points or lists.
            Each paragraph should be 3-5 sentences.
            """

            try:
                response = self.generate_content(prompt)
                formatted_text = self.reformat_for_readability(response.text)

                categorized_risks[category] = {
                    "analysis": formatted_text,
                    "legal_references": category_legal_refs,
                    "rag_ref_count": ref_count,
                    "timestamp": datetime.now().isoformat()
                }
                time.sleep(1)
            except Exception as e:
                print(f"    [WARN] Error: {e}")
                categorized_risks[category] = {
                    "analysis": f"Classification unavailable: {str(e)}",
                    "legal_references": [],
                    "rag_ref_count": 0,
                    "timestamp": datetime.now().isoformat()
                }

        return categorized_risks

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Executive summary  ── UNCHANGED
    # ─────────────────────────────────────────────────────────────────────────
    def generate_risk_summary(self, categorized_risks: Dict, sale_deed_text: str) -> str:
        """Generate executive summary grounded in risk classification output"""
        print("\n[INFO] Generating risk summary...")

        rag_coverage = {
            cat: data.get('rag_ref_count', 0)
            for cat, data in categorized_risks.items()
        }

        slim_risks = {
            cat: {"analysis": data["analysis"], "timestamp": data["timestamp"]}
            for cat, data in categorized_risks.items()
        }

        prompt = f"""
        You are a senior legal partner writing an executive summary for a client.

        RISK ANALYSIS (RAG-grounded — legal provisions retrieved per category):
        {json.dumps(slim_risks, indent=2, ensure_ascii=False)}

        RAG COVERAGE (number of legal provisions retrieved per risk category):
        {json.dumps(rag_coverage, indent=2)}

        Write a comprehensive executive summary in 5-6 clear paragraphs:

        First paragraph: Provide the overall risk rating (High Risk/Medium Risk/Low Risk)
        and explain what this property transaction analysis covered. Mention that the
        analysis is grounded in retrieved legal provisions from a curated knowledge base.

        Second paragraph: Describe the 3-5 most critical issues identified. For each,
        explain what it is, which law or regulation it relates to, and why it matters.

        Third paragraph: Provide a clear transaction recommendation (Proceed / Proceed
        with Caution / Do Not Proceed / Further Investigation Required) with rationale.

        Fourth paragraph: Outline priority actions — immediate (within 7 days), short
        term (within 30 days), and before registration.

        Fifth paragraph: Discuss financial implications — costs for remediation,
        compliance, potential penalties, and legal exposure.

        Final paragraph: Address timeline — how long to resolve issues and the impact
        on the transaction timeline.

        Write in flowing professional paragraphs. 4-6 sentences each. No bullet points.
        """

        try:
            response = self.generate_content(prompt)
            return self.reformat_for_readability(response.text)
        except Exception as e:
            return f"Summary generation failed: {e}"

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Key findings for first-page summary  — FULLY ALGORITHMIC
    # No Gemini call.  Extracts findings from data already in memory:
    #   • lri_obj  — threat references with matched text + page/line locations
    #   • categorized_risks — Gemini's written analysis (already generated)
    #   • gemini_severity_label — parsed by regex in the LRI engine
    # ─────────────────────────────────────────────────────────────────────────
    def generate_key_findings(
        self,
        categorized_risks: Dict,
        lri_obj: "LegalRiskIndex",
    ) -> List[Dict]:
        """
        Builds the cover-page key findings list WITHOUT any extra Gemini call.

        Strategy per category:
          1. Severity  — taken directly from lri_obj.categories[cat].gemini_severity_label
             (already regex-parsed from Gemini's written analysis).
          2. Threat bullets — each detected threat from lri_obj gives one bullet:
               "<pattern label> (Page <n>, Line <m>)"
          3. Analysis sentences — if threat count < 2, we mine Gemini's written
             analysis for sentences that contain negative signal words and use
             those as additional bullets (trimmed to ≤ 120 chars).
          4. Categories with severity None/N/A and zero threats are omitted.

        Returns list of dicts: [{"category", "severity", "findings"}, ...]
        sorted by severity (High → Medium → Low).
        """
        import re as _re

        print("\n[INFO] Extracting key findings algorithmically (no Gemini call)...")

        # Negative-signal words that mark a sentence as a problem finding
        _NEGATIVE = _re.compile(
            r'\b(not\s+obtained|missing|absent|unpaid|outstanding|arrear|'
            r'unauthorized|violation|non.?compli|pending|injunction|stay\s+order|'
            r'encumbrance|dispute|unclear|not\s+clear|not\s+registered|'
            r'not\s+established|not\s+verified|deficien|under.?paid|'
            r'acquisition|demolition|illegal|invalid|challenged|contested|'
            r'criminal|attachment|lis\s+pendens|gazette)\b',
            _re.IGNORECASE
        )

        _SEV_ORDER = {"High": 0, "Medium": 1, "Low": 2, "None": 3, "N/A": 4}

        result = []

        for cat, risk_data in categorized_risks.items():
            lri_cat = lri_obj.categories.get(cat) if lri_obj else None
            sev_label = (lri_cat.gemini_severity_label
                         if lri_cat else "N/A")

            # Skip categories Gemini found clean
            if sev_label.lower() in ("none", "n/a"):
                continue

            findings = []

            # ── Source 1: threat references from pattern scanner ──────────
            if lri_cat and lri_cat.threats:
                for t in lri_cat.threats:
                    bullet = (
                        f"{t.pattern_matched} "
                        f"(Page {t.page_number}, Line {t.line_start})"
                    )
                    findings.append(bullet)

            # ── Source 2: negative sentences from Gemini's written analysis
            # Used when threats < 3 to ensure at least 2-3 bullets per category
            if len(findings) < 3:
                analysis_text = risk_data.get("analysis", "")
                # Split into sentences on ". " or ".\n"
                sentences = _re.split(r'(?<=[.!?])\s+', analysis_text)
                for sent in sentences:
                    sent = sent.strip()
                    # Must contain a negative signal and be a reasonable length
                    if (20 < len(sent) < 180
                            and _NEGATIVE.search(sent)
                            and sent not in findings):
                        # Trim to 120 chars at a word boundary
                        if len(sent) > 120:
                            trimmed = sent[:120].rsplit(' ', 1)[0]
                            sent = trimmed.rstrip('.,;') + '…'
                        findings.append(sent)
                    if len(findings) >= 4:
                        break

            # ── Deduplicate while preserving order ────────────────────────
            seen, unique = set(), []
            for f in findings:
                key = f[:60].lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(f)
            findings = unique[:5]   # cap at 5 bullets per category

            if not findings:
                continue

            result.append({
                "category": cat,
                "severity": sev_label,
                "findings": findings,
            })

        # Sort: High first, then Medium, then Low
        result.sort(key=lambda x: _SEV_ORDER.get(x["severity"], 99))

        print(f"[OK] Key findings extracted: {len(result)} category/categories with issues "
              f"({sum(len(r['findings']) for r in result)} total bullets)")
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS  ── UNCHANGED
    # ─────────────────────────────────────────────────────────────────────────
    def reformat_for_readability(self, text: str) -> str:
        """Convert AI-generated structured text into flowing, readable paragraphs"""
        if not text or len(text.strip()) == 0:
            return ""

        text = text.replace('**', '').replace('###', '').replace('####', '').replace('##', '').replace('`', '')

        lines = text.split('\n')
        paragraphs = []
        current_paragraph = []
        in_list = False

        for line in lines:
            line = line.strip()
            if not line:
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                in_list = False
                continue

            if line.startswith('- ') or line.startswith('* ') or line.startswith('• '):
                line = line[2:].strip()
                in_list = True
            elif len(line) > 2 and line[0].isdigit() and line[1] == '.':
                line = line[line.index('.')+1:].strip()
                in_list = True
            elif len(line) > 3 and line[0].isdigit() and line[1].isdigit() and line[2] == '.':
                line = line[line.index('.')+1:].strip()
                in_list = True

            if ':' in line and (line.split(':')[0].isupper() or line.split(':')[0].istitle()):
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                header_text = line.split(':')[0].strip()
                content_after = line.split(':', 1)[1].strip() if len(line.split(':')) > 1 else ''
                paragraphs.append(f"{header_text}: {content_after}" if content_after else header_text)
                current_paragraph = []
                continue

            if in_list:
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                current_paragraph.append(line)
            else:
                current_paragraph.append(line)

        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))

        cleaned_paragraphs = []
        for para in paragraphs:
            para = ' '.join(para.split())
            if para and not para[0].isupper():
                para = para[0].upper() + para[1:]
            if para and para[-1] not in '.!?':
                para += '.'
            if len(para) > 10:
                cleaned_paragraphs.append(para)

        return '\n\n'.join(cleaned_paragraphs)

    def extract_sale_deed_content(self, document_path: str) -> str:
        """Extract text from PDF or Word document"""
        print(f"\n[INFO] Extracting content from: {document_path}")
        file_extension = Path(document_path).suffix.lower()

        if file_extension == '.pdf':
            import fitz
            doc = fitz.open(document_path)
            text = "".join(page.get_text() for page in doc)
            doc.close()
        elif file_extension in ['.docx', '.doc']:
            from docx import Document
            doc = Document(document_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        print(f"[OK] Extracted {len(text)} characters")
        return text

    def translate_batch_fast(self, texts: List[str], target_language: str) -> List[str]:
        """Fast translation using deep-translator"""
        if target_language == "English":
            return texts

        lang_code = self.LANGUAGES[target_language]["code"]
        print(f"\n[INFO] Translating {len(texts)} items to {target_language}...")

        translated_texts = []
        success_count = 0

        for i, text in enumerate(texts):
            if not text or not isinstance(text, str) or not text.strip():
                translated_texts.append("")
                continue
            try:
                translator = GoogleTranslator(source='en', target=lang_code)
                text_to_translate = text.strip()
                if len(text_to_translate) > 4500:
                    chunks = [text_to_translate[j:j+4500] for j in range(0, len(text_to_translate), 4500)]
                    translated_text = ' '.join(translator.translate(c) for c in chunks)
                else:
                    translated_text = translator.translate(text_to_translate)
                translated_texts.append(translated_text)
                success_count += 1
                if (i + 1) % 5 == 0:
                    print(f"  [OK] Translated {i+1}/{len(texts)} items")
                time.sleep(0.1)
            except Exception as e:
                print(f"  [WARN] Error on item {i+1}: {str(e)[:50]}")
                translated_texts.append(text)

        print(f"[OK] Translation complete! {success_count}/{len(texts)} items successfully translated")
        return translated_texts

    def setup_fonts_for_language(self, language: str):
        """Register fonts for Indian languages"""
        font_name = self.LANGUAGES[language]["font"]
        if font_name and font_name not in pdfmetrics.getRegisteredFontNames():
            try:
                font_path = f"./fonts/{font_name}-Regular.ttf"
                if os.path.exists(font_path):
                    pdfmetrics.registerFont(TTFont(font_name, font_path))
                    print(f"[OK] Registered font: {font_name}")
                else:
                    print(f"[WARN] Font not found: {font_path}")
                    return None
            except Exception as e:
                print(f"[WARN] Font error: {e}")
                return None
        return font_name

    # ─────────────────────────────────────────────────────────────────────────
    # PDF REPORT  ── ORIGINAL LOGIC INTACT; LRI section appended at the end
    # ─────────────────────────────────────────────────────────────────────────
    def generate_pdf_report(
        self,
        sale_deed_path: str,
        rag_analysis: Dict,
        categorized_risks: Dict,
        risk_summary: str,
        output_path: str,
        language: str = "English",
        lri_obj: LegalRiskIndex = None,
        key_findings: List[Dict] = None,      # cover-page bullet findings
    ):
        """Generate PDF report. All original sections are preserved;
        key findings appear on the cover page, LRI section is appended at end."""
        print(f"\n[INFO] Generating PDF report...")

        font_name = self.setup_fonts_for_language(language)
        if not font_name:
            font_name = "Helvetica"

        # ── Build translation list (UNCHANGED) ────────────────────────────
        base_strings = [
            "Legal Due Diligence Report",
            "Property Sale Deed Analysis",
            "Property Document",
            "Report Date",
            "Report Language",
            "Analysis Engine",
            "Executive Summary",
            risk_summary,
            "Detailed Risk Analysis by Category",
            "Legal Provisions Referenced",
            "Supporting Legal References",
        ]

        if language != "English":
            to_translate = list(base_strings)

            for category in categorized_risks.keys():
                to_translate.append(category)
            for risk_data in categorized_risks.values():
                to_translate.append(risk_data.get('analysis', ''))
            for aspect, data in rag_analysis.items():
                to_translate.append(aspect)
                to_translate.append(data['analysis'])

            translated = self.translate_batch_fast(to_translate, language)

            idx = 0
            title_main              = translated[idx]; idx += 1
            title_sub               = translated[idx]; idx += 1
            prop_doc_label          = translated[idx]; idx += 1
            report_date_label       = translated[idx]; idx += 1
            report_lang_label       = translated[idx]; idx += 1
            analysis_eng_label      = translated[idx]; idx += 1
            exec_summary_heading    = translated[idx]; idx += 1
            risk_summary_trans      = translated[idx]; idx += 1
            detail_risk_heading     = translated[idx]; idx += 1
            legal_prov_heading      = translated[idx]; idx += 1
            support_ref_heading     = translated[idx]; idx += 1

            category_translations = {}
            for category in categorized_risks.keys():
                category_translations[category] = translated[idx]; idx += 1

            risk_translations = {}
            for category in categorized_risks.keys():
                risk_translations[category] = translated[idx]; idx += 1

            aspect_translations = {}
            analysis_translations = {}
            for aspect in rag_analysis.keys():
                aspect_translations[aspect]   = translated[idx]; idx += 1
                analysis_translations[aspect] = translated[idx]; idx += 1
        else:
            title_main              = "Legal Due Diligence Report"
            title_sub               = "Property Sale Deed Analysis"
            prop_doc_label          = "Property Document"
            report_date_label       = "Report Date"
            report_lang_label       = "Report Language"
            analysis_eng_label      = "Analysis Engine"
            exec_summary_heading    = "Executive Summary"
            risk_summary_trans      = risk_summary
            detail_risk_heading     = "Detailed Risk Analysis by Category"
            legal_prov_heading      = "Legal Provisions Referenced"
            support_ref_heading     = "Supporting Legal References"
            category_translations   = {k: k for k in categorized_risks.keys()}
            risk_translations       = {k: v.get('analysis', '') for k, v in categorized_risks.items()}
            aspect_translations     = {k: k for k in rag_analysis.keys()}
            analysis_translations   = {k: v['analysis'] for k, v in rag_analysis.items()}

        # ── Build PDF ─────────────────────────────────────────────────────
        doc = SimpleDocTemplate(
            output_path, pagesize=A4,
            rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50
        )
        story = []
        styles = getSampleStyleSheet()

        def _font(bold=False):
            if font_name != "Helvetica":
                return font_name
            return "Helvetica-Bold" if bold else "Helvetica"

        title_style = ParagraphStyle('TitleStyle', parent=styles['Title'],
            fontSize=22, textColor=colors.HexColor('#1a5490'), spaceAfter=6,
            alignment=TA_CENTER, fontName=_font(True), leading=26)

        subtitle_style = ParagraphStyle('SubtitleStyle', parent=styles['Heading2'],
            fontSize=14, textColor=colors.HexColor('#555555'), spaceAfter=30,
            alignment=TA_CENTER, fontName=_font(), leading=18)

        heading1_style = ParagraphStyle('Heading1Style', parent=styles['Heading1'],
            fontSize=16, textColor=colors.HexColor('#1a5490'), spaceAfter=12,
            spaceBefore=20, fontName=_font(True), leading=20)

        heading2_style = ParagraphStyle('Heading2Style', parent=styles['Heading2'],
            fontSize=13, textColor=colors.HexColor('#2c5aa0'), spaceAfter=10,
            spaceBefore=15, fontName=_font(True), leading=16)

        heading3_style = ParagraphStyle('Heading3Style', parent=styles['Heading2'],
            fontSize=11, textColor=colors.HexColor('#3a6b9e'), spaceAfter=6,
            spaceBefore=10, fontName=_font(True), leading=14)

        body_style = ParagraphStyle('BodyStyle', parent=styles['BodyText'],
            fontSize=10, alignment=TA_JUSTIFY, spaceAfter=12,
            fontName=_font(), leading=15, firstLineIndent=0)

        ref_style = ParagraphStyle('RefStyle', parent=styles['BodyText'],
            fontSize=9, textColor=colors.HexColor('#444444'), spaceAfter=6,
            fontName=_font(), leading=13, leftIndent=12,
            borderPadding=(4, 4, 4, 4))

        # ── TITLE PAGE (UNCHANGED) ─────────────────────────────────────────
        story.append(Spacer(1, 1*inch))
        story.append(Paragraph(title_main, title_style))
        story.append(Paragraph(title_sub, subtitle_style))
        story.append(Spacer(1, 0.5*inch))

        info_data = [
            [prop_doc_label + ":",     Path(sale_deed_path).name],
            [report_date_label + ":",  datetime.now().strftime("%d %B %Y, %I:%M %p")],
            [report_lang_label + ":",  language],
            [analysis_eng_label + ":", "AI + RAG System"],
        ]
        info_table = Table(info_data, colWidths=[2.2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f0f4f8')),
            ('TEXTCOLOR', (0,0), (0,-1), colors.HexColor('#1a5490')),
            ('TEXTCOLOR', (1,0), (1,-1), colors.black),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (0,-1), _font(True)),
            ('FONTNAME', (1,0), (1,-1), _font()),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 12),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),
            ('LEFTPADDING', (0,0), (-1,-1), 12),
            ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#d0d8e0'))
        ]))
        story.append(info_table)

        # ── KEY FINDINGS — cover page highlight box ────────────────────────
        if key_findings:
            story.append(Spacer(1, 0.3*inch))

            # Severity colour map
            sev_hex = {
                "High":     "#c62828",
                "Medium":   "#e65100",
                "Low":      "#2e7d32",
                "None":     "#555555",
            }
            sev_bg = {
                "High":     "#fff5f5",
                "Medium":   "#fff8f0",
                "Low":      "#f5fff7",
                "None":     "#f7f7f7",
            }
            sev_badge_bg = {
                "High":     "#c62828",
                "Medium":   "#e65100",
                "Low":      "#2e7d32",
                "None":     "#777777",
            }

            # Section heading
            kf_heading_style = ParagraphStyle(
                'KFHeading', parent=styles['Heading1'],
                fontSize=13, textColor=colors.HexColor('#1a5490'),
                spaceAfter=6, spaceBefore=0,
                fontName=_font(True), leading=16
            )
            kf_subhead_style = ParagraphStyle(
                'KFSubHead', parent=styles['Normal'],
                fontSize=8, textColor=colors.HexColor('#666666'),
                spaceAfter=8, fontName=_font(), leading=11
            )
            kf_cat_style = ParagraphStyle(
                'KFCat', parent=styles['Normal'],
                fontSize=9, textColor=colors.HexColor('#1a5490'),
                spaceAfter=2, spaceBefore=0,
                fontName=_font(True), leading=12
            )
            kf_bullet_style = ParagraphStyle(
                'KFBullet', parent=styles['Normal'],
                fontSize=9, textColor=colors.HexColor('#222222'),
                spaceAfter=2, fontName=_font(), leading=13,
                leftIndent=14, firstLineIndent=-8
            )

            story.append(Paragraph("Key Findings at a Glance", kf_heading_style))
            story.append(Paragraph(
                "Critical issues identified in this document — see detailed sections for full analysis.",
                kf_subhead_style
            ))

            for item in key_findings:
                cat   = item["category"]
                sev   = item.get("severity", "Medium")
                finds = item.get("findings", [])
                if not finds:
                    continue

                bg_color  = colors.HexColor(sev_bg.get(sev, "#f7f7f7"))
                cat_color = colors.HexColor(sev_hex.get(sev, "#333333"))
                badge_bg  = colors.HexColor(sev_badge_bg.get(sev, "#777777"))

                # Category row: name + severity badge side by side
                badge_text = f"  {sev.upper()}  "
                badge_para = Paragraph(
                    f'<font color="white"><b>{badge_text}</b></font>',
                    ParagraphStyle('Badge', parent=styles['Normal'],
                                   fontSize=8, fontName=_font(True),
                                   leading=10, alignment=TA_CENTER)
                )
                cat_para = Paragraph(
                    f'<font color="{sev_hex.get(sev, "#333333")}"><b>{cat}</b></font>',
                    ParagraphStyle('CatLabel', parent=styles['Normal'],
                                   fontSize=10, fontName=_font(True),
                                   leading=13)
                )

                # Header row for this category
                header_row = Table(
                    [[cat_para, badge_para]],
                    colWidths=[4.6*inch, 0.8*inch]
                )
                header_row.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), bg_color),
                    ('BACKGROUND', (1, 0), (1, 0), badge_bg),
                    ('ALIGN',      (0, 0), (0, 0), 'LEFT'),
                    ('ALIGN',      (1, 0), (1, 0), 'CENTER'),
                    ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
                    ('TOPPADDING',    (0, 0), (-1, -1), 5),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                    ('LEFTPADDING',   (0, 0), (0, 0), 8),
                    ('RIGHTPADDING',  (1, 0), (1, 0), 4),
                    ('LINEBELOW', (0, 0), (-1, 0), 0.5,
                     colors.HexColor(sev_hex.get(sev, "#aaaaaa"))),
                ]))
                story.append(header_row)

                # Bullet rows
                bullet_rows = []
                for finding in finds:
                    bullet_rows.append([
                        Paragraph(
                            f'<font color="#555555">•</font>  {finding}',
                            kf_bullet_style
                        )
                    ])

                bullets_table = Table(bullet_rows, colWidths=[5.4*inch])
                bullets_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), bg_color),
                    ('TOPPADDING',    (0, 0), (-1, -1), 3),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                    ('LEFTPADDING',   (0, 0), (-1, -1), 8),
                    ('LINEBELOW', (0, -1), (-1, -1), 0.5,
                     colors.HexColor('#d8dee6')),
                ]))
                story.append(bullets_table)
                story.append(Spacer(1, 0.06*inch))

        story.append(PageBreak())

        # ── EXECUTIVE SUMMARY (UNCHANGED) ─────────────────────────────────
        story.append(Paragraph(exec_summary_heading, heading1_style))
        story.append(Spacer(1, 0.1*inch))
        for para in risk_summary_trans.split('\n\n'):
            if para.strip():
                story.append(Paragraph(para.strip(), body_style))
        story.append(PageBreak())

        # ── DETAILED RISK ANALYSIS (UNCHANGED) ────────────────────────────
        story.append(Paragraph(detail_risk_heading, heading1_style))
        story.append(Spacer(1, 0.15*inch))

        for cat_idx, (category, risk_data) in enumerate(categorized_risks.items(), 1):
            ref_count = risk_data.get('rag_ref_count', 0)
            header_text = (
                f"{cat_idx}. {category_translations.get(category, category)}"
                f"  —  RAG provisions used: {ref_count}"
            )
            story.append(Paragraph(header_text, heading2_style))
            story.append(Spacer(1, 0.08*inch))

            for para in risk_translations[category].split('\n\n'):
                if para.strip():
                    story.append(Paragraph(para.strip(), body_style))

            category_refs = risk_data.get('legal_references', [])
            if category_refs:
                story.append(Spacer(1, 0.08*inch))
                story.append(Paragraph(legal_prov_heading, heading3_style))
                for ref in category_refs:
                    src = ref.get('source', 'Unknown')
                    content_preview = ref.get('content', '')[:300].strip()
                    if content_preview and not content_preview.endswith('.'):
                        content_preview += '...'
                    story.append(Paragraph(
                        f"<b>[{src}]</b>  {content_preview}",
                        ref_style
                    ))

            if cat_idx < len(categorized_risks):
                story.append(Spacer(1, 0.2*inch))

        story.append(PageBreak())

        # ── SUPPORTING LEGAL REFERENCES (UNCHANGED) ───────────────────────
        story.append(Paragraph(support_ref_heading, heading1_style))
        story.append(Spacer(1, 0.15*inch))

        for ref_idx, (aspect, data) in enumerate(rag_analysis.items(), 1):
            story.append(Paragraph(
                f"{ref_idx}. {aspect_translations[aspect].title()}",
                heading2_style
            ))
            story.append(Spacer(1, 0.08*inch))

            for para in analysis_translations[aspect].split('\n\n'):
                if para.strip():
                    story.append(Paragraph(para.strip(), body_style))

            story.append(Spacer(1, 0.15*inch))

        # ── NEW: LRI + THREAT INVENTORY SECTION (appended after existing sections)
        if lri_obj is not None:
            story.append(PageBreak())
            self._build_lri_pdf_section(story, lri_obj, heading1_style,
                                         heading2_style, heading3_style,
                                         body_style, ref_style, _font)

        try:
            doc.build(story)
            print(f"[OK] PDF report generated: {output_path}")
            print(f"   Language: {language}")
            print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")
        except Exception as e:
            print(f"[ERROR] PDF generation error: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # NEW: build LRI section for PDF
    # ─────────────────────────────────────────────────────────────────────────
    def _build_lri_pdf_section(
        self, story, lri_obj: LegalRiskIndex,
        h1, h2, h3, body, ref_style, font_fn
    ):
        """
        Adds two sub-sections to the PDF story (in-place):
          A) Legal Risk Index — score, band, category breakdown table
          B) Threat Inventory — per-category threat count + document references
        All computed by the algorithm; Gemini is NOT involved here.
        """
        styles = getSampleStyleSheet()

        # colour map for band
        band_hex = {
            "Low":      "#2e7d32",
            "Medium":   "#f57c00",
            "High":     "#d84315",
            "Critical": "#b71c1c",
        }
        band_color_hex = band_hex.get(lri_obj.band, "#333333")

        lri_title_style = ParagraphStyle(
            'LRITitle', parent=styles['Heading1'],
            fontSize=16, textColor=colors.HexColor('#1a5490'),
            spaceAfter=6, spaceBefore=10,
            fontName=font_fn(True), leading=20
        )
        band_style = ParagraphStyle(
            'BandStyle', parent=styles['Normal'],
            fontSize=13, textColor=colors.HexColor(band_color_hex),
            spaceAfter=4, fontName=font_fn(True), leading=16
        )
        label_style = ParagraphStyle(
            'LabelStyle', parent=styles['Normal'],
            fontSize=9, textColor=colors.HexColor('#555555'),
            spaceAfter=2, fontName=font_fn(), leading=12
        )
        mono_style = ParagraphStyle(
            'MonoStyle', parent=styles['Normal'],
            fontSize=8, textColor=colors.HexColor('#333333'),
            spaceAfter=3, fontName=font_fn(), leading=12,
            leftIndent=8
        )

        # ── A) LRI SCORE SECTION ──────────────────────────────────────────
        story.append(Paragraph("Legal Risk Index (LRI)", lri_title_style))
        story.append(Paragraph(
            "Computed algorithmically from keyword density, RAG similarity scores, "
            "and threat density — independent of the Gemini language model.",
            ParagraphStyle('Desc', parent=styles['Normal'], fontSize=9,
                           textColor=colors.HexColor('#666666'),
                           fontName=font_fn(), leading=13, spaceAfter=10)
        ))

        # Score + band card as a table
        score_data = [
            ["Overall LRI Score", f"{lri_obj.lri_score:.1f} / 100"],
            ["Risk Band",         lri_obj.band.upper()],
            ["Assessment",        lri_obj.band_description],
            ["Total Threats",     str(lri_obj.total_threats)],
            ["Document Words",    f"{lri_obj.word_count:,}"],
        ]
        score_table = Table(score_data, colWidths=[2.0*inch, 4.2*inch])
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f7f9fc')),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor(band_color_hex)),
            ('TEXTCOLOR',  (0, 1), (-1, 1), colors.white),
            ('FONTNAME',   (0, 0), (0, -1), font_fn(True)),
            ('FONTNAME',   (1, 0), (1, -1), font_fn()),
            ('FONTSIZE',   (0, 0), (-1, -1), 10),
            ('ALIGN',      (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d0d8e0')),
        ]))
        story.append(score_table)
        story.append(Spacer(1, 0.2*inch))

        # Category breakdown table
        story.append(Paragraph("Category Breakdown  (LRI = Σ wᵢ × Pᵢ × 100)", h3))
        story.append(Spacer(1, 0.06*inch))

        cat_header = ["Category", "Weight (wᵢ)", "Probability (Pᵢ)", "Contribution", "Threats"]
        cat_rows = [cat_header]
        for cat_name, cat in lri_obj.categories.items():
            cat_rows.append([
                cat_name,
                f"{cat.weight:.0%}",
                f"{cat.probability:.1%}",
                f"{cat.weighted_contribution * 100:.2f}",
                str(cat.threat_count),
            ])
        cat_rows.append(["", "", "TOTAL LRI", f"{lri_obj.lri_score:.2f}", ""])

        cat_table = Table(cat_rows, colWidths=[2.1*inch, 0.9*inch, 1.1*inch, 1.0*inch, 0.7*inch])
        cat_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5490')),
            ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
            ('FONTNAME',   (0, 0), (-1, 0), font_fn(True)),
            ('FONTNAME',   (0, 1), (-1, -1), font_fn()),
            ('FONTSIZE',   (0, 0), (-1, -1), 9),
            ('ALIGN',      (1, 0), (-1, -1), 'CENTER'),
            ('ALIGN',      (0, 0), (0, -1), 'LEFT'),
            ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('ROWBACKGROUNDS', (0, 1), (-1, -2),
             [colors.HexColor('#f7f9fc'), colors.HexColor('#eef2f7')]),
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#e8edf5')),
            ('FONTNAME',   (0, -1), (-1, -1), font_fn(True)),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#c8d4e0')),
        ]))
        story.append(cat_table)
        story.append(Spacer(1, 0.1*inch))

        # Sub-score detail table
        story.append(Paragraph(
            "Sub-score Detail  (KD = keyword density · RS = RAG similarity · TD = threat density · GS = Gemini severity)",
            ParagraphStyle('SmallLabel', parent=styles['Normal'], fontSize=8,
                           textColor=colors.HexColor('#666666'),
                           fontName=font_fn(), leading=11, spaceAfter=4)
        ))
        sub_header = ["Category", "KD (30%)", "RS (25%)", "TD (15%)", "GS (30%)", "GS Label", "Pᵢ"]
        sub_rows = [sub_header]
        for cat_name, cat in lri_obj.categories.items():
            sub_rows.append([
                cat_name,
                f"{cat.keyword_density:.1%}",
                f"{cat.rag_similarity_score:.1%}",
                f"{cat.threat_density_score:.1%}",
                f"{cat.gemini_severity_score:.1%}",
                cat.gemini_severity_label,
                f"{cat.probability:.1%}",
            ])
        sub_table = Table(sub_rows, colWidths=[1.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.65*inch, 0.65*inch])
        sub_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
            ('FONTNAME',   (0, 0), (-1, 0), font_fn(True)),
            ('FONTNAME',   (0, 1), (-1, -1), font_fn()),
            ('FONTSIZE',   (0, 0), (-1, -1), 8),
            ('ALIGN',      (1, 0), (-1, -1), 'CENTER'),
            ('ALIGN',      (0, 0), (0, -1), 'LEFT'),
            ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.HexColor('#f7f9fc'), colors.HexColor('#eef2f7')]),
            ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#c8d4e0')),
        ]))
        story.append(sub_table)

        # ── B) THREAT INVENTORY SECTION ───────────────────────────────────
        story.append(PageBreak())
        story.append(Paragraph("Threat Inventory — Document References", lri_title_style))
        story.append(Paragraph(
            "Each entry below was detected by pattern matching on the deed text. "
            "Page numbers and line numbers are as they appear in the source document, "
            "allowing you to locate and verify each finding directly.",
            ParagraphStyle('Desc2', parent=styles['Normal'], fontSize=9,
                           textColor=colors.HexColor('#666666'),
                           fontName=font_fn(), leading=13, spaceAfter=12)
        ))

        threat_h2_style = ParagraphStyle(
            'ThreatH2', parent=styles['Heading2'],
            fontSize=12, textColor=colors.HexColor('#2c5aa0'),
            spaceAfter=6, spaceBefore=12,
            fontName=font_fn(True), leading=15
        )
        threat_ref_style = ParagraphStyle(
            'ThreatRef', parent=styles['Normal'],
            fontSize=8, textColor=colors.HexColor('#555555'),
            spaceAfter=1, fontName=font_fn(), leading=11, leftIndent=12
        )
        threat_ctx_style = ParagraphStyle(
            'ThreatCtx', parent=styles['Normal'],
            fontSize=8, textColor=colors.HexColor('#333333'),
            spaceAfter=6, fontName=font_fn(), leading=12, leftIndent=12,
            borderPadding=(3, 3, 3, 3)
        )

        for cat_name, cat in lri_obj.categories.items():
            story.append(Paragraph(
                f"{cat_name}  —  {cat.threat_count} threat(s) found",
                threat_h2_style
            ))

            if not cat.threats:
                story.append(Paragraph(
                    "No threat indicators detected for this category.",
                    ParagraphStyle('NoThreat', parent=styles['Normal'], fontSize=9,
                                   textColor=colors.HexColor('#888888'),
                                   fontName=font_fn(), leading=12, spaceAfter=8)
                ))
                continue

            # Build a mini-table per category: one row per threat
            threat_table_data = [
                ["ID", "Threat", "Page", "Lines", "Matched Text"]
            ]
            for t in cat.threats:
                matched_short = t.matched_text[:60] + ("…" if len(t.matched_text) > 60 else "")
                threat_table_data.append([
                    t.threat_id,
                    t.pattern_matched,
                    str(t.page_number),
                    f"{t.line_start}–{t.line_end}",
                    matched_short,
                ])

            t_col_widths = [0.55*inch, 1.8*inch, 0.45*inch, 0.6*inch, 2.4*inch]
            t_table = Table(threat_table_data, colWidths=t_col_widths)
            t_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3a6b9e')),
                ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
                ('FONTNAME',   (0, 0), (-1, 0), font_fn(True)),
                ('FONTNAME',   (0, 1), (-1, -1), font_fn()),
                ('FONTSIZE',   (0, 0), (-1, -1), 8),
                ('ALIGN',      (2, 0), (3, -1), 'CENTER'),
                ('ALIGN',      (0, 0), (1, -1), 'LEFT'),
                ('ALIGN',      (4, 0), (4, -1), 'LEFT'),
                ('VALIGN',     (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('LEFTPADDING', (0, 0), (-1, -1), 5),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                 [colors.HexColor('#f9fbfc'), colors.HexColor('#eef2f7')]),
                ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#c8d4e0')),
                ('WORDWRAP', (0, 0), (-1, -1), True),
            ]))
            story.append(t_table)

            # Context snippets below the table
            for t in cat.threats:
                ctx_short = t.context_snippet[:250] + ("…" if len(t.context_snippet) > 250 else "")
                story.append(Paragraph(
                    f"[{t.threat_id}] context: {ctx_short}",
                    threat_ctx_style
                ))

            story.append(Spacer(1, 0.1*inch))

        # Formula footer
        story.append(Spacer(1, 0.2*inch))
        formula_data = [
            ["Formula", "LRI = Σ (wᵢ × Pᵢ) × 100"],
            ["Pᵢ weights", "KD 30%  ·  RS 25%  ·  TD 15%  ·  GS (Gemini severity) 30%"],
            ["Domain weights", "Ownership 30%  ·  Litigation 25%  ·  Tax 20%  ·  Construction 15%  ·  Government 10%"],
            ["GS source", "Severity label (High/Medium/Low/None) parsed from Gemini's written analysis by regex"],
        ]
        f_table = Table(formula_data, colWidths=[1.3*inch, 5.0*inch])
        f_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0f4f8')),
            ('FONTNAME',   (0, 0), (0, -1), font_fn(True)),
            ('FONTNAME',   (1, 0), (1, -1), font_fn()),
            ('FONTSIZE',   (0, 0), (-1, -1), 8),
            ('TEXTCOLOR',  (0, 0), (-1, -1), colors.HexColor('#333366')),
            ('ALIGN',      (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#c8d4e0')),
        ]))
        story.append(f_table)

    # ─────────────────────────────────────────────────────────────────────────
    # PIPELINE  ── Only change: run LRI engine before Gemini steps,
    #              pass lri_obj to generate_pdf_report, save JSON sidecar.
    # ─────────────────────────────────────────────────────────────────────────
    def process_sale_deed(
        self,
        sale_deed_path: str,
        output_language: str = "English",
        output_dir: str = "./legal_reports"
    ) -> str:
        """Complete end-to-end processing"""
        print("=" * 70)
        print("[INFO] RAG-Enhanced Legal Due Diligence System")
        print("=" * 70)

        os.makedirs(output_dir, exist_ok=True)

        # Step 0: extract text (unchanged)
        sale_deed_text = self.extract_sale_deed_content(sale_deed_path)

        # Steps 1–3: Gemini analysis (completely unchanged — runs first)
        rag_analysis      = self.analyze_sale_deed_with_rag(sale_deed_text)
        categorized_risks = self.classify_risks(rag_analysis, sale_deed_text)
        risk_summary      = self.generate_risk_summary(categorized_risks, sale_deed_text)

        # ── FEATURE 1 + 2 (NEW) ───────────────────────────────────────────
        # Runs AFTER Gemini so that categorized_risks (Gemini's per-category
        # written analysis) can be used as the 4th sub-score in P_i.
        # The severity label (High/Medium/Low/None) is extracted from Gemini's
        # text by regex — no additional LLM call is made here.
        print("\n[INFO] Running algorithmic risk engine (LRI + threat scan)...")
        real_page_breaks = _get_pdf_page_breaks(sale_deed_path)

        lri_obj, lri_json_str = run_risk_engine(
            text=sale_deed_text,
            real_page_breaks=real_page_breaks,
            vectorizer=self.vectorizer,
            legal_db=self.legal_db,
            rag_queries_per_category=self.RISK_CATEGORY_RAG_QUERIES,
            categorized_risks=categorized_risks,   # Gemini analysis fed in here
            print_cli=True,       # prints full threat inventory + LRI to stdout
            return_json=True,
        )

        # Save LRI JSON sidecar (useful for the web backend)
        if lri_json_str:
            timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path   = os.path.join(output_dir, f"lri_{timestamp}.json")
            with open(json_path, "w", encoding="utf-8") as fj:
                fj.write(lri_json_str)
            print(f"[OK] LRI JSON saved: {json_path}")
        # ── END FEATURE 1 + 2 ─────────────────────────────────────────────

        # ── KEY FINDINGS for cover page ────────────────────────────────────
        key_findings = self.generate_key_findings(categorized_risks, lri_obj)
        # ── END KEY FINDINGS ──────────────────────────────────────────────

        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        lang_code   = self.LANGUAGES[output_language]['code']
        output_path = os.path.join(output_dir, f"legal_due_diligence_{timestamp}_{lang_code}.pdf")

        # PDF generation: lri_obj is passed so the new section is appended
        self.generate_pdf_report(
            sale_deed_path=sale_deed_path,
            rag_analysis=rag_analysis,
            categorized_risks=categorized_risks,
            risk_summary=risk_summary,
            output_path=output_path,
            language=output_language,
            lri_obj=lri_obj,
            key_findings=key_findings,
        )

        print("\n" + "=" * 70)
        print("[OK] Analysis complete!")
        print(f"[INFO] Report: {output_path}")
        print("=" * 70)
        return output_path


def main():
    """Main execution"""
    print("\n[INFO] Legal Due Diligence System - RAG Enhanced")
    print("=" * 70)

    try:
        system = LegalDueDiligenceSystem(
            db_path="./vector_db",
            collection_name="pdf_collection"
        )
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return

    print("\n[INFO] Enter path to sale deed document (PDF or DOCX):")
    sale_deed_path = input("Path: ").strip().strip('"').strip("'")

    if not os.path.exists(sale_deed_path):
        print(f"[ERROR] File not found: {sale_deed_path}")
        return

    print("\n[INFO] Choose output language:")
    for i, lang in enumerate(system.LANGUAGES.keys(), 1):
        print(f"{i}. {lang}")

    lang_choice = input("\nEnter choice (1-7): ").strip()

    try:
        output_language = list(system.LANGUAGES.keys())[int(lang_choice) - 1]
    except (ValueError, IndexError):
        print("Invalid choice. Using English.")
        output_language = "English"

    try:
        report_path = system.process_sale_deed(
            sale_deed_path=sale_deed_path,
            output_language=output_language,
            output_dir="./legal_reports"
        )
        print(f"\n[OK] Success! Report: {report_path}")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

