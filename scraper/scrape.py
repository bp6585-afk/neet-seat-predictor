"""
Scraper for neetugguidance.in — NEET 2025 closing ranks.
Handles 2-row colspan headers (the main quirk of this site).

Table layouts found:
  - Gujarat / Rajasthan college tables : rows=colleges, cols=Category(colspan=2)→AIR/Score
  - Rajasthan / AIQ category tables    : rows=categories, cols=Round(colspan=2)→AIR/Score
  - Karnataka                          : rows=colleges,   cols=Round(colspan=2)→AIR/Score
  - Tamil Nadu                         : rows=colleges,   cols=categories → Score only

Run:  python -m scraper.scrape
"""

import re
import time
import requests
import pandas as pd

from bs4 import BeautifulSoup
from pathlib import Path

BASE_URL  = "https://www.neetugguidance.in/state-institute.php"
HOME_URL  = "https://www.neetugguidance.in/"
DELAY_SEC = 8   # polite delay between pages — needed to avoid Mod_Security drops

# Shared session — must call _init_session() before any _get() calls
_SESSION = None  # type: requests.Session


def _init_session():
    global _SESSION
    _SESSION = requests.Session()
    _SESSION.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    })
    # Warm-up — establishes cookies and sets Referer for subsequent requests
    _SESSION.get(HOME_URL, timeout=30)
    _SESSION.headers["Referer"] = HOME_URL
    time.sleep(2)
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Category + Gender canonicalisation ────────────────────────────────────
# Maps raw column code → (canonical_category, gender)
# gender ∈ {"Male", "Female", "All"}
#
# Rajasthan uses B/G suffixes (URB=Unreserved Boys, URG=Unreserved Girls, etc.)
# All other states in this source don't break down by gender → "All"
CATEGORY_GENDER_MAP = {
    # ── Gender-neutral codes (All states except Rajasthan) ─────────────
    "ur":       ("General", "All"),  "open":     ("General", "All"),
    "oc":       ("General", "All"),  "gen":      ("General", "All"),
    "gm":       ("General", "All"),  "opn":      ("General", "All"),
    "oth":      ("General", "All"),  "gmp":      ("General", "All"),
    "obc":      ("OBC",     "All"),  "se":       ("OBC",     "All"),
    "sebc":     ("OBC",     "All"),  "bc":       ("OBC",     "All"),
    "bcm":      ("OBC",     "All"),  "mbc":      ("OBC",     "All"),
    "mbc& dnc": ("OBC",     "All"),  "mbc&dnc":  ("OBC",     "All"),
    "ews":      ("EWS",     "All"),
    "sc":       ("SC",      "All"),  "sca":      ("SC",      "All"),
    "st":       ("ST",      "All"),  "sa":       ("ST",      "All"),
    "nri":      ("NRI",     "All"),
    "management":("Management","All"), "mgmt":   ("Management","All"),
    # ── Rajasthan Boys (B suffix) ──────────────────────────────────────
    "urb":  ("General", "Male"), "obb":  ("OBC",  "Male"),
    "ewb":  ("EWS",     "Male"), "mbb":  ("OBC",  "Male"),
    "scb":  ("SC",      "Male"), "stb":  ("ST",   "Male"),
    "sab":  ("ST",      "Male"),
    # ── Rajasthan Girls (G suffix) ─────────────────────────────────────
    "urg":  ("General", "Female"), "obg":  ("OBC",  "Female"),
    "ewg":  ("EWS",     "Female"), "mbg":  ("OBC",  "Female"),
    "scg":  ("SC",      "Female"), "stg":  ("ST",   "Female"),
    "sag":  ("ST",      "Female"),
}

def _canon(raw_code: str):
    """Return (category, gender) for a raw column code."""
    key = raw_code.strip().lower()
    return CATEGORY_GENDER_MAP.get(key, (raw_code.title(), "All"))

# Keep a plain category-only lookup for backwards compat
CATEGORY_CANON = {k: v[0] for k, v in CATEGORY_GENDER_MAP.items()}

ROUND_CANON = {
    "r1": "Round 1", "round 1": "Round 1", "round1": "Round 1",
    "r2": "Round 2", "round 2": "Round 2", "round2": "Round 2",
    "r3": "Mop-up Round", "r4": "Stray Round", "r5": "Stray Round",
    "mop up": "Mop-up Round", "mop up round": "Mop-up Round",
    "mopup": "Mop-up Round", "mop-up": "Mop-up Round",
    "stray": "Stray Round", "stray round": "Stray Round",
}

# Approximate NEET score → All India Rank (2025, General category)
_SCORE_RANK = [
    (720, 1),      (700, 50),     (690, 150),    (680, 350),
    (670, 800),    (660, 1500),   (650, 2500),   (640, 4000),
    (630, 6000),   (620, 8500),   (615, 11000),  (610, 14000),
    (605, 17500),  (600, 21500),  (595, 26500),  (590, 32000),
    (585, 38500),  (580, 46000),  (575, 55000),  (570, 66000),
    (565, 79000),  (560, 94000),  (555, 111000), (550, 130000),
    (545, 152000), (540, 176000), (535, 202000), (530, 230000),
    (525, 260000), (520, 292000), (510, 360000), (500, 430000),
    (490, 502000), (480, 576000), (450, 700000),
]

def _score_to_rank(score: int) -> int:
    """Interpolate NEET score to approximate All India Rank."""
    for i, (s, r) in enumerate(_SCORE_RANK):
        if score >= s:
            if i == 0:
                return r
            s_high, r_high = _SCORE_RANK[i - 1]
            frac = (score - s) / max(s_high - s, 1)
            return int(r + frac * (r_high - r))
    return 700_000


# ── Core HTML helpers ──────────────────────────────────────────────────────

def _get(coldesc_id: int, retries: int = 3) -> BeautifulSoup:
    # Fresh session each time — server drops persistent connections after ~3 reqs
    last_err = None
    for attempt in range(retries):
        try:
            _init_session()
            resp = _SESSION.get(
                BASE_URL,
                params={"colser_id": 7, "coldesc_id": coldesc_id},
                timeout=30,
            )
            resp.raise_for_status()
            time.sleep(DELAY_SEC)
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as exc:
            last_err = exc
            wait = DELAY_SEC * (attempt + 2)
            print(f"    attempt {attempt+1} failed ({exc}), retrying in {wait}s…")
            time.sleep(wait)
    raise last_err


def _parse_table(tbl):
    """
    Parse a <table> handling 2-row colspan+rowspan headers.

    The site mixes two patterns:
      - colspan=2 on round/category headers  (expand horizontally)
      - rowspan=2 on label columns            (cell absent from row1)

    Strategy:
      1. Expand row0 cells by their colspan into `expanded_row0`.
      2. Count row0 cells that have rowspan=2 → prepend that many
         empty slots to row1 so positions align again.
      3. If row1 now looks like a sub-header (contains AIR/Score),
         combine: "Round1" + "AIR" → "Round1_AIR".
      4. Use the resulting columns; skip rows whose length doesn't match.
    """
    rows = tbl.find_all("tr")
    if len(rows) < 2:
        return None

    def cell_text(td):
        return td.get_text(" ", strip=True)

    row0_cells = rows[0].find_all(["th", "td"])

    # Build expanded_row0 and count rowspan=2 cells
    expanded_row0 = []
    n_rowspan = 0
    for td in row0_cells:
        text   = cell_text(td)
        cspan  = int(td.get("colspan", 1))
        rspan  = int(td.get("rowspan", 1))
        if rspan > 1:
            n_rowspan += cspan          # these columns are ABSENT from row1
        expanded_row0.extend([text] * cspan)

    row1_texts = [cell_text(td) for td in rows[1].find_all(["th", "td"])] if len(rows) > 1 else []

    # Prepend empty strings for rowspan cells so row1 aligns with expanded_row0
    aligned_row1 = [""] * n_rowspan + row1_texts

    sub_header_keywords = {"air", "score", "rank", "marks"}
    is_sub_header = (
        len(aligned_row1) == len(expanded_row0)
        and any(t.lower() in sub_header_keywords for t in aligned_row1 if t.strip())
    )

    if is_sub_header:
        cols = []
        for parent, child in zip(expanded_row0, aligned_row1):
            p, c = parent.strip(), child.strip()
            if c and p and p.lower() not in c.lower():
                cols.append(f"{p}_{c}")
            elif c:
                cols.append(c)
            else:
                cols.append(p if p else f"col{len(cols)}")
        data_start = 2
    else:
        cols       = [cell_text(td) for td in row0_cells]
        data_start = 1

    # Collect data rows — pad/truncate so every row matches len(cols)
    records = []
    for row in rows[data_start:]:
        cells = [cell_text(td) for td in row.find_all(["td", "th"])]
        if not cells:
            continue
        if len(cells) < len(cols):
            cells += [""] * (len(cols) - len(cells))
        elif len(cells) > len(cols):
            cells = cells[:len(cols)]
        records.append(cells)

    if not records:
        return None

    return pd.DataFrame(records, columns=cols)


def _first_int(text: str):
    """Return the first integer from a string like '21190 / 534' or '21,190'."""
    nums = re.findall(r"\d[\d,]*", text.replace(",", ""))
    return int(nums[0]) if nums else None


def _col(name: str) -> str:
    return name.strip().lower()


# ── Gujarat ────────────────────────────────────────────────────────────────
#  Structure: rows=colleges, cols=Category(colspan=2)→AIR/Score
#  Two tables: State Quota, Management Quota
#  Cols after parse: sno, College Name, OPEN_AIR, OPEN_Score,
#                    EWS_AIR, EWS_Score, SE_AIR, SE_Score,
#                    SC_AIR, SC_Score, ST_AIR, ST_Score

def scrape_gujarat(year: int = 2025) -> pd.DataFrame:
    soup = _get(13)
    records = []
    quota_labels = ["State Quota", "Management Quota"]

    for tbl_idx, tbl in enumerate(soup.find_all("table")):
        df = _parse_table(tbl)
        if df is None or len(df) < 2:
            continue

        cols = [_col(c) for c in df.columns]
        # College name is usually the second column
        college_col = next(
            (df.columns[i] for i, c in enumerate(cols)
             if "college" in c or "name" in c),
            df.columns[1] if len(df.columns) > 1 else None,
        )
        if college_col is None:
            continue

        # Find *_AIR columns → these hold the closing rank per category
        air_cols = {c: df.columns[i]
                    for i, c in enumerate(cols)
                    if c.endswith("_air") or (c.endswith("air") and "_" in c)}

        if not air_cols:
            continue

        quota = quota_labels[tbl_idx] if tbl_idx < len(quota_labels) else "State Quota"

        for _, row in df.iterrows():
            college = str(row[college_col]).strip()
            if not college or college.lower() in ("college name", "nan", "name", ""):
                continue
            for raw_col, orig_col in air_cols.items():
                cat_key          = raw_col.replace("_air", "").strip()
                category, gender = _canon(cat_key)
                rank             = _first_int(str(row[orig_col]))
                if rank and rank > 0:
                    records.append({
                        "college": college, "state": "Gujarat", "year": year,
                        "round": "Stray Round", "quota": quota,
                        "category": category, "gender": gender,
                        "closing_rank": rank,
                    })

    df_out = pd.DataFrame(records).drop_duplicates()
    df_out.to_csv(RAW_DIR / "gujarat.csv", index=False)
    print(f"  Gujarat : {len(df_out)} records")
    return df_out


# ── Rajasthan ──────────────────────────────────────────────────────────────
#  Page has TWO layouts:
#    A) Category-row tables (aggregate round-wise closing rank per category)
#       cols: category | Round1_AIR | Round1_Score | Round2_AIR | ...
#    B) College-row tables (college-wise per-category closing rank)
#       cols: sno | college | UR_AIR | OBC_AIR | EWS_AIR | MBC_AIR | SC_AIR | ST_AIR | SA_AIR
#
#  We use BOTH: A → round-wise cutoffs (mapped to "Rajasthan Pool")
#               B → college-wise

RAJASTHAN_CATS = {"ur", "urb", "urg", "obb", "obg", "obc", "ewb", "ewg", "ews",
                  "mbb", "mbg", "mbc", "scb", "scg", "sc", "stb", "stg", "st",
                  "sab", "sag", "sa"}
RAJASTHAN_ROUNDS = {"round 1", "round1", "round 2", "round2",
                    "mop up", "mop up round", "mopup", "stray", "stray round"}

def scrape_rajasthan(year: int = 2025) -> pd.DataFrame:
    soup = _get(30)
    records = []

    for tbl in soup.find_all("table"):
        df = _parse_table(tbl)
        if df is None or len(df) < 2:
            continue
        cols = [_col(c) for c in df.columns]

        # ── Layout A: first col is a category code ─────────────────────
        first_vals = [_col(str(v)) for v in df.iloc[:, 0].dropna()]
        is_cat_rows = sum(1 for v in first_vals if v in RAJASTHAN_CATS) > len(first_vals) * 0.4

        if is_cat_rows:
            # Find round_AIR columns: "round 1_air", "r1_air", etc.
            round_air_cols = {c: df.columns[i]
                              for i, c in enumerate(cols)
                              if c.endswith("_air") and
                              any(k in c for k in ("round", "mop", "stray", "r1", "r2", "r3", "r4"))}
            for _, row in df.iterrows():
                cat_key          = _col(str(row.iloc[0]))
                category, gender = _canon(cat_key)
                for round_col, orig_col in round_air_cols.items():
                    round_key  = round_col.replace("_air", "").strip()
                    round_name = ROUND_CANON.get(round_key, "Round 1")
                    rank = _first_int(str(row[orig_col]))
                    if rank and rank > 0:
                        records.append({
                            "college": "Rajasthan State Quota Pool",
                            "state": "Rajasthan", "year": year,
                            "round": round_name, "quota": "State Quota",
                            "category": category, "gender": gender,
                            "closing_rank": rank,
                        })

        # ── Layout B: second col contains college names ─────────────────
        else:
            college_col = next(
                (df.columns[i] for i, c in enumerate(cols)
                 if "college" in c or "name" in c),
                df.columns[1] if len(df.columns) > 1 else None,
            )
            if college_col is None:
                continue

            air_cols = {c: df.columns[i]
                        for i, c in enumerate(cols)
                        if c.endswith("_air")}
            if not air_cols:
                continue

            for _, row in df.iterrows():
                college = str(row[college_col]).strip()
                if not college or college.lower() in ("college name", "nan", "name", ""):
                    continue
                for raw_col, orig_col in air_cols.items():
                    cat_key          = raw_col.replace("_air", "").strip()
                    category, gender = _canon(cat_key)
                    rank             = _first_int(str(row[orig_col]))
                    if rank and rank > 0:
                        records.append({
                            "college": college, "state": "Rajasthan", "year": year,
                            "round": "Stray Round", "quota": "State Quota",
                            "category": category, "gender": gender,
                            "closing_rank": rank,
                        })

    df_out = pd.DataFrame(records).drop_duplicates()
    df_out.to_csv(RAW_DIR / "rajasthan.csv", index=False)
    print(f"  Rajasthan: {len(df_out)} records")
    return df_out


# ── Tamil Nadu ─────────────────────────────────────────────────────────────
#  Structure: rows=colleges, cols=OC | BC | BCM | MBC&DNC | SC | ST | SCA
#  Values are SCORES (not ranks) — converted via _score_to_rank()

TN_CAT_COLS = {"oc", "bc", "bcm", "mbc& dnc", "mbc&dnc", "mbc", "sc", "st", "sca"}

def scrape_tamil_nadu(year: int = 2025) -> pd.DataFrame:
    soup = _get(32)
    records = []
    round_idx = 0
    round_names = ["Round 1", "Round 2", "Mop-up Round"]

    for tbl in soup.find_all("table"):
        df = _parse_table(tbl)
        if df is None or len(df) < 2:
            continue
        cols = [_col(c) for c in df.columns]

        # Check if any column matches TN category names
        cat_matches = [c for c in cols if c in TN_CAT_COLS]
        if not cat_matches:
            continue

        round_name = round_names[min(round_idx, len(round_names) - 1)]
        round_idx += 1

        college_col = next(
            (df.columns[i] for i, c in enumerate(cols)
             if "college" in c or "name" in c),
            df.columns[1] if len(df.columns) > 1 else None,
        )
        if college_col is None:
            continue

        for _, row in df.iterrows():
            college = str(row[college_col]).strip()
            if not college or college.lower() in ("college name", "nan", "colleges", ""):
                continue
            for raw_cat in cat_matches:
                orig_col = df.columns[cols.index(raw_cat)]
                val = str(row[orig_col]).strip()
                num = _first_int(val)
                if num and num > 0:
                    # TN values ≤ 720 are scores; convert to approximate rank
                    if num <= 720:
                        closing_rank = _score_to_rank(num)
                    else:
                        closing_rank = num  # already a rank
                    category, gender = _canon(raw_cat)
                    records.append({
                        "college": college, "state": "Tamil Nadu", "year": year,
                        "round": round_name, "quota": "State Quota",
                        "category": category, "gender": gender,
                        "closing_rank": closing_rank,
                    })

    df_out = pd.DataFrame(records).drop_duplicates()
    df_out.to_csv(RAW_DIR / "tamil_nadu.csv", index=False)
    print(f"  Tamil Nadu: {len(df_out)} records")
    return df_out


# ── Karnataka ─────────────────────────────────────────────────────────────
#  Structure: rows=colleges, cols=Round(colspan=2)→AIR/Score
#  Multiple tables by quota: GM, GMP, OPN, OTH, NRI
#  Cols after parse: sno | college | estb | seats | fee | Round1_AIR | Round2_AIR | MopUp_AIR

KA_QUOTA_KEYWORDS = {
    "gm quota": ("General", "State Quota"),
    "gmp quota": ("General", "Management Quota"),
    "opn quota": ("General", "AIQ"),
    "oth quota": ("General", "AIQ"),
    "nri quota": ("NRI", "NRI"),
    " gm ": ("General", "State Quota"),
    "govt quota": ("General", "State Quota"),
    "management": ("General", "Management Quota"),
    "private": ("General", "Management Quota"),
}

def scrape_karnataka(year: int = 2025) -> pd.DataFrame:
    soup = _get(18)
    records = []

    for tbl in soup.find_all("table"):
        df = _parse_table(tbl)
        if df is None or len(df) < 2:
            continue
        cols = [_col(c) for c in df.columns]

        # Find Round_AIR columns
        round_air_cols = {c: df.columns[i]
                          for i, c in enumerate(cols)
                          if c.endswith("_air") and
                          any(k in c for k in ("round", "mop", "stray", "r1", "r2"))}
        if not round_air_cols:
            continue

        # Infer quota/category from the heading preceding this table
        prev_text = ""
        for sib in tbl.find_all_previous(["h2", "h3", "h4", "p", "b", "strong"], limit=3):
            prev_text = sib.get_text(" ", strip=True).lower()
            if prev_text:
                break

        category, quota = "General", "State Quota"
        for kw, (cat, q) in KA_QUOTA_KEYWORDS.items():
            if kw in prev_text:
                category, quota = cat, q
                break

        college_col = next(
            (df.columns[i] for i, c in enumerate(cols)
             if "college" in c or "name" in c),
            df.columns[1] if len(df.columns) > 1 else None,
        )
        if college_col is None:
            continue

        for _, row in df.iterrows():
            college = str(row[college_col]).strip()
            if not college or college.lower() in ("college", "colleges", "nan", "name", ""):
                continue
            for raw_col, orig_col in round_air_cols.items():
                round_key = raw_col.replace("_air", "").strip()
                round_name = ROUND_CANON.get(round_key, "Round 1")
                rank = _first_int(str(row[orig_col]))
                if rank and rank > 0:
                    records.append({
                        "college": college, "state": "Karnataka", "year": year,
                        "round": round_name, "quota": quota,
                        "category": category, "gender": "All",
                        "closing_rank": rank,
                    })

    df_out = pd.DataFrame(records).drop_duplicates()
    df_out.to_csv(RAW_DIR / "karnataka.csv", index=False)
    print(f"  Karnataka: {len(df_out)} records")
    return df_out


# ── All India Quota ────────────────────────────────────────────────────────
#  Structure: rows=categories, cols=Round(colspan=2)→AIR/Score
#  Cols: category | R1_AIR | R1_Score | R2_AIR | R2_Score | R3_AIR | R4_AIR | R5_AIR

INST_URL = "https://www.neetugguidance.in/institutes.php"

# Women-only institutions — their seats are Female-only
_WOMEN_ONLY = {"lady hardinge", "bps government medical college for women",
               "bps gmcw", "svims-spmcw"}

# Category mapping for central-institution pages (UR/GN → General etc.)
_INST_CAT = {
    "ur": "General", "gn": "General", "open": "General", "gen": "General",
    "obc": "OBC", "ews": "EWS", "sc": "SC", "st": "ST",
}


def _get_inst(coldesc_id: int, retries: int = 3) -> BeautifulSoup:
    """Fresh-session fetch for institutes.php (central institutions)."""
    last_err = None
    for attempt in range(retries):
        try:
            _init_session()
            time.sleep(DELAY_SEC)
            resp = _SESSION.get(
                INST_URL,
                params={"colser_id": 7, "coldesc_id": coldesc_id},
                timeout=30,
            )
            resp.raise_for_status()
            if len(resp.text) < 1000:
                raise ValueError(f"Response too short: {len(resp.text)} bytes")
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as exc:
            last_err = exc
            wait = DELAY_SEC * (attempt + 2)
            print(f"    inst/{coldesc_id} attempt {attempt+1} failed ({exc}), retry in {wait}s…")
            time.sleep(wait)
    raise last_err


def _parse_aiims_table(tbl, year):
    """
    Parse the 'ALL 20 AIIMS college-wise' table from the AIIMS page.
    Row 0:  S.No. | ALL 20 AIIMS | Estb | UR(colspan=2) | OBC | EWS | SC | ST
    Row 1:  (empty×3)             |      | Rank | Score   | …
    Data:   serial | college_name | estb | ur_rank | ur_score | …
    Returns list of record dicts (one per college×category).
    """
    df = _parse_table(tbl)
    if df is None or len(df) < 3:
        return []

    cols_lower = [_col(c) for c in df.columns]

    # Identify the college-name column (contains "aiims" in header)
    name_col = None
    for i, c in enumerate(cols_lower):
        if "aiims" in c or "name" in c or "institute" in c:
            name_col = df.columns[i]
            break
    if name_col is None and len(df.columns) > 1:
        name_col = df.columns[1]   # fallback: second column

    # Collect *_rank or *_air columns and map to category
    rank_cols = {}
    for i, c in enumerate(cols_lower):
        if not (c.endswith("_rank") or c.endswith("_air")):
            continue
        raw_cat = c.replace("_rank", "").replace("_air", "").strip()
        canon = _INST_CAT.get(raw_cat)
        if canon:
            rank_cols[canon] = df.columns[i]

    records = []
    for _, row in df.iterrows():
        college = str(row[name_col]).strip()
        if not college or college.lower() in ("name", "nan", "college", ""):
            continue
        gender = "Female" if any(w in college.lower() for w in _WOMEN_ONLY) else "All"
        for cat, col in rank_cols.items():
            rank = _first_int(str(row[col]))
            if rank and rank > 0:
                records.append({
                    "college": college, "state": "All India", "year": year,
                    "round": "Stray Round", "quota": "AIQ",
                    "category": cat, "gender": gender, "closing_rank": rank,
                })
    return records


def _parse_inst_college_table(tbl, year):
    """
    Parse one per-college table from Central/JIPMER/AFMC/ESIC pages.

    Row 0:  CollegeName | [ROUND] | UR(colspan=2) | OBC | EWS | SC | ST
    Row 1:  QUOTA       | [''   ] | AIR | Score    | …
     OR
    Row 0:  CollegeName | UR | OBC | EWS | SC | ST
    Row 1:  QUOTA | ROUND | AIR | Score | …
    Data:   quota_name | [round] | air | score | …
    """
    rows = tbl.find_all("tr")
    if len(rows) < 3:
        return []

    r0_cells = rows[0].find_all(["th", "td"])
    if not r0_cells:
        return []

    # College name = first cell, strip leading "1. " numbering
    college_raw = r0_cells[0].get_text(" ", strip=True)
    college = re.sub(r"^\d+[\.\)]\s*", "", college_raw).strip()
    if not college or len(college) < 8:
        return []
    # Reject tables that look like aggregate/category tables not college tables
    if college.upper() in ("CATEGORY", "QUOTA", "S.NO.", "SL.NO.", "NAME", "COLLEGE NAME"):
        return []

    gender = "Female" if any(w in college.lower() for w in _WOMEN_ONLY) else "All"

    # Detect ROUND column in either row
    r0_texts = [c.get_text(strip=True).upper() for c in r0_cells]
    r1_cells = rows[1].find_all(["th", "td"])
    r1_texts = [c.get_text(strip=True).upper() for c in r1_cells]
    has_round = "ROUND" in r0_texts or "ROUND" in r1_texts

    # Build category list from row0 (skip college-name cell and ROUND cell)
    categories = []
    for c in r0_cells[1:]:
        txt = c.get_text(strip=True).upper()
        if txt in ("ROUND", ""):
            continue
        canon = _INST_CAT.get(txt.lower())
        if canon:
            span = int(c.get("colspan", 2))
            categories.extend([canon] * max(1, span // 2))

    if not categories:
        return []

    records = []
    for row in rows[2:]:
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
        if not cells or not cells[0] or cells[0].upper() in ("QUOTA", ""):
            continue

        quota_raw = cells[0].lower()
        # Keep only open/all-india AIQ seats; skip IP, NRI, Paid, Management etc.
        # If quota text is short/ambiguous (e.g. just "AIQ"), keep it.
        if len(quota_raw) > 8 and "all india" not in quota_raw and "open" not in quota_raw:
            continue

        idx = 1
        round_name = "Stray Round"
        if has_round and len(cells) > 1:
            rnd_raw = _col(cells[1])
            # Only treat cells[1] as a round if it looks like one (not a pure number)
            if rnd_raw in ROUND_CANON or re.match(r"^r\d$|^round", rnd_raw):
                round_name = ROUND_CANON.get(rnd_raw, rnd_raw.title())
                idx = 2

        data = cells[idx:]
        for i, cat in enumerate(categories):
            if i * 2 < len(data):
                rank = _first_int(data[i * 2])
                if rank and rank > 0:
                    records.append({
                        "college": college, "state": "All India", "year": year,
                        "round": round_name, "quota": "AIQ",
                        "category": cat, "gender": gender, "closing_rank": rank,
                    })
    return records


def _parse_category_col_table(tbl, year):
    """
    Parse tables where rows=categories, cols=CollegeName × Year.
    Used for JIPMER/AFMC pages.

    Row 0: Category | CollegeA MBBS 2025 Last Rank | CollegeA MBBS 2024… | CollegeB…
    Row 1: AIR | Score | AIR | Score | ...
    Data: GN | 260 | 629 | ...
    """
    df = _parse_table(tbl)
    if df is None or len(df) < 2:
        return []

    cols_lower = [_col(c) for c in df.columns]

    # Find the category column (first col = "category" or values look like cat codes)
    cat_col = df.columns[0]
    first_vals = {_col(str(v)) for v in df[cat_col].dropna()}
    if not first_vals.intersection({"gn", "ur", "obc", "ews", "sc", "st",
                                     "general", "unreserved"}):
        return []

    # Find AIR columns that mention the target year
    year_str = str(year)
    air_cols = {}  # college_name → orig_col
    for i, c in enumerate(cols_lower):
        if year_str not in c:
            continue
        if not (c.endswith("_air") or c.endswith("_rank") or "last rank" in c):
            continue
        # Extract college name from the full column header
        orig_col = df.columns[i]
        # Strip the _AIR/_Rank/_Score suffix first, then clean junk words
        base = re.sub(r"_(air|rank|score)$", "", orig_col, flags=re.IGNORECASE)
        college = re.sub(
            r"(mbbs|bds|last\s*rank|cutoff|\d{4}|round\s*\d|\s*vs\s*.*)",
            "", base, flags=re.IGNORECASE,
        ).strip().strip("_").strip()
        if college:
            air_cols[college] = orig_col

    if not air_cols:
        return []

    records = []
    for _, row in df.iterrows():
        cat_raw = _col(str(row[cat_col]))
        cat = _INST_CAT.get(cat_raw)
        if not cat:
            continue
        for college, col in air_cols.items():
            rank = _first_int(str(row[col]))
            if rank and rank > 0:
                gender = "Female" if any(w in college.lower() for w in _WOMEN_ONLY) else "All"
                records.append({
                    "college": college.strip(), "state": "All India", "year": year,
                    "round": "Stray Round", "quota": "AIQ",
                    "category": cat, "gender": gender, "closing_rank": rank,
                })
    return records


def scrape_aiq(year: int = 2025) -> pd.DataFrame:
    """
    Scrapes per-college AIQ closing ranks from neetugguidance.in/institutes.php:
      coldesc_id=42 → 20 AIIMS campuses
      coldesc_id=39 → Central Institutes (MAMC, UCMS, Lady Hardinge, VMMC, BHU, AMU…)
      coldesc_id=43 → JIPMER (Puducherry + Karaikal)
      coldesc_id=40 → AFMC Pune
      coldesc_id=41 → ESIC Medical Colleges
      coldesc_id=44 → CMC Vellore
    """
    records = []

    # ── AIIMS: college-wise table ──────────────────────────────────────────
    try:
        soup = _get_inst(42)
        for tbl in soup.find_all("table"):
            rows = tbl.find_all("tr")
            # College-wise table has many rows and "S.No." or "AIIMS" in header
            if len(rows) < 10:
                continue
            r0_txt = " ".join(c.get_text(strip=True).lower()
                              for c in rows[0].find_all(["th", "td"]))
            if "s.no" in r0_txt or "aiims" in r0_txt:
                parsed = _parse_aiims_table(tbl, year)
                if parsed:
                    records.extend(parsed)
                    break   # only need the college-wise table
    except Exception as exc:
        print(f"  AIQ/AIIMS failed: {exc}")

    # ── Central, JIPMER, AFMC, ESIC, CMC — per-college tables ─────────────
    for cid, label in [(39, "Central"), (43, "JIPMER"), (40, "AFMC"),
                       (41, "ESIC"), (44, "CMC Vellore")]:
        try:
            soup = _get_inst(cid)
            page_records = []
            for tbl in soup.find_all("table"):
                r0 = tbl.find("tr")
                if not r0:
                    continue
                first_cell_txt = r0.find(["th", "td"])
                if not first_cell_txt:
                    continue
                txt = first_cell_txt.get_text(strip=True).upper()

                if txt in ("CATEGORY", "UR", "GN", "OPEN"):
                    # JIPMER/AFMC format: rows=categories, cols=colleges
                    parsed = _parse_category_col_table(tbl, year)
                elif txt not in ("QUOTA", "S.NO.", "SL.NO.", "NAME"):
                    # Standard per-college format
                    parsed = _parse_inst_college_table(tbl, year)
                else:
                    parsed = []
                page_records.extend(parsed)
            print(f"  AIQ/{label}: {len(page_records)} records")
            records.extend(page_records)
        except Exception as exc:
            print(f"  AIQ/{label} failed: {exc}")

    df_out = pd.DataFrame(records).drop_duplicates() if records else pd.DataFrame(
        columns=["college", "state", "year", "round", "quota", "category", "gender", "closing_rank"]
    )
    df_out.to_csv(RAW_DIR / "aiq.csv", index=False)
    print(f"  AIQ total: {len(df_out)} records")
    return df_out


# ── Merge all ──────────────────────────────────────────────────────────────

def scrape_all() -> pd.DataFrame:
    print("Starting scrape …")
    # AIQ last — first request in a session is most likely to drop; warm up with states first
    scrapers = [
        ("Karnataka",  scrape_karnataka),
        ("Gujarat",    scrape_gujarat),
        ("Rajasthan",  scrape_rajasthan),
        ("Tamil Nadu", scrape_tamil_nadu),
        ("AIQ",        scrape_aiq),
    ]
    dfs = []
    for name, fn in scrapers:
        try:
            df = fn()
            dfs.append(df)
        except Exception as e:
            print(f"  {name} FAILED: {e}")

    if not dfs:
        raise RuntimeError("All scrapers failed — check network connectivity.")

    unified = pd.concat(dfs, ignore_index=True)

    # ── Clean ─────────────────────────────────────────────────────────────
    unified["closing_rank"] = pd.to_numeric(unified["closing_rank"], errors="coerce")
    unified = unified.dropna(subset=["closing_rank"])
    unified["closing_rank"] = unified["closing_rank"].astype(int)

    # Drop nonsensical ranks
    unified = unified[(unified["closing_rank"] >= 1) & (unified["closing_rank"] <= 800_000)]

    # Keep only canonical categories (drop parsing artifacts like 'R2', 'Ri', etc.)
    valid_cats = {"General", "OBC", "EWS", "SC", "ST", "NRI", "Management"}
    unified = unified[unified["category"].isin(valid_cats)]

    # Drop aggregate pool placeholders — not real colleges
    unified = unified[~unified["college"].isin(["Rajasthan State Quota Pool", "AIQ National Pool"])]

    # Drop rows where college name looks like a parsing artifact
    # (contains newlines, multiple "College Name" tokens, or is just numbers)
    unified = unified[~unified["college"].str.contains(r"\n|College Name.*College Name", regex=True, na=False)]
    unified = unified[unified["college"].str.len() > 5]
    unified = unified[~unified["college"].str.match(r"^\d+$", na=False)]

    # Normalise college names — strip extra whitespace
    unified["college"] = unified["college"].str.strip().str.replace(r"\s+", " ", regex=True)

    # For AIQ central institutions, keep minimum closing rank per combination.
    # Multiple seat pools (IP Quota sub-types etc.) may have leaked through;
    # the minimum represents the true open-seat competitive cutoff.
    aiq_mask = unified["state"] == "All India"
    key = ["college", "state", "year", "round", "category", "gender"]
    aiq_deduped = (
        unified[aiq_mask]
        .sort_values("closing_rank")
        .drop_duplicates(subset=key, keep="first")
    )
    unified = pd.concat([unified[~aiq_mask], aiq_deduped], ignore_index=True)

    # Keep only core columns (drop tn_score etc.)
    core_cols = ["college", "state", "year", "round", "quota", "category", "gender", "closing_rank"]
    unified = unified[[c for c in core_cols if c in unified.columns]]

    out = PROCESSED_DIR / "unified.csv"
    unified.to_csv(out, index=False)
    print(f"\nDone. {len(unified)} rows → {out}")
    print(unified.groupby("state")["college"].count().to_string())
    return unified


if __name__ == "__main__":
    scrape_all()
