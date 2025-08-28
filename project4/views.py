import os, json, csv, time, uuid
from io import BytesIO

import numpy as np
import pandas as pd

from django.http import HttpResponse
from django.shortcuts import render, redirect

# ReportLab (Platypus) for documentation PDF
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem,
    HRFlowable, Table, TableStyle
)

# Recommender utilities you already have
from .recommender import (
    train_model,
    load_model,
    compute_new_user_profile,
    recommend,
)

# ------------------------------------------------------------------------------------
# One-time model & metadata loading
# ------------------------------------------------------------------------------------
APP_DIR    = os.path.dirname(__file__)
DATA_DIR   = os.path.join(APP_DIR, 'data')
MODEL_DIR  = os.path.join(APP_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'svd.npz')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Train once if missing
if not os.path.exists(MODEL_PATH):
    train_model(n_components=20)

# Load factors + metadata (keep names consistent with your recommender)
U_all, Sigma, VT, user_ids, movie_ids, movie_titles = load_model()

# Build a fast id->index map for movies (used when computing predictions)
MOVIE_INDEX = {int(mid): i for i, mid in enumerate(list(movie_ids))}

# Movie metadata (genres) if present
MOVIES_CSV = os.path.join(DATA_DIR, 'movies.csv')
if os.path.exists(MOVIES_CSV):
    movies_df = pd.read_csv(MOVIES_CSV)
    movie_genres = dict(zip(movies_df.movieId, movies_df.genres))
else:
    movies_df = pd.DataFrame(columns=['movieId', 'title', 'genres'])
    movie_genres = {}

# ------------------------------------------------------------------------------------
# Study constants + unified logging schema
# ------------------------------------------------------------------------------------
STOP_AFTER_RATINGS = 8
LOG_PATH = os.path.join(DATA_DIR, 'study_log.csv')

# Use ONE fixed schema for every log row, regardless of event type
FIELDNAMES = [
    'event', 'pid', 'condition', 'pilot',
    # consent
    'agreed',
    # pre/post survey
    'trust', 'usefulness', 'expertise', 'effort',
    # rating event
    'movie_id', 'movie_title', 'rating', 'rt_ms',
    'confidence', 'info_gain', 'top5_current', 'top5_predicted',
]

# ------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------
def _now_ms():
    return int(time.time() * 1000)

def _get_pid(request):
    """Anonymous participant id persisted in session."""
    if 'pid' not in request.session:
        request.session['pid'] = str(uuid.uuid4())
    return request.session['pid']

def _get_condition(request):
    """Between-subjects randomization: 'guided' vs 'control' once per participant."""
    if 'condition' not in request.session:
        import random
        request.session['condition'] = random.choice(['guided', 'control'])
    return request.session['condition']

def _is_pilot(request):
    """Mark this run pilot if ?pilot=1 was used once."""
    if 'pilot' not in request.session:
        request.session['pilot'] = 1 if request.GET.get('pilot') == '1' else 0
    return request.session['pilot']

def _log_event(row: dict):
    """Append a row to CSV with a fixed header/schema for all events.
       Flush + fsync so the next request sees it immediately.
    """
    new = not os.path.exists(LOG_PATH)
    safe = {k: '' for k in FIELDNAMES}
    for k, v in row.items():
        if k in safe:
            safe[k] = v  # ignore unexpected keys to keep schema stable

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    with open(LOG_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_MINIMAL)
        if new:
            writer.writeheader()
        writer.writerow(safe)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass

# ------------------------------------------------------------------------------------
# Landing + Documentation PDF
# ------------------------------------------------------------------------------------
def index(request):
    """Landing: links to documentation, start study, etc."""
    return render(request, 'project4/index.html')

def download_doc(request):
    """
    Generate a short, practical write-up for Project 4.
    Covers: representation/model, guidance strategy, interface & study design, and data/ethics.
    ASCII-safe wording (no special math symbols).
    """
    buf = BytesIO()

    # document + styles
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        rightMargin=48, leftMargin=48, topMargin=60, bottomMargin=54
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, spaceAfter=12))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="H3", parent=styles["Heading3"], spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], leading=14, spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, textColor=colors.grey))

    story = []

    # title
    story += [
        Paragraph("Project 4: Guided Cold-Start Recommender", styles["TitleCenter"]),
        Paragraph("Implementation notes and study plan", styles["Small"]),
        HRFlowable(width="100%", thickness=0.5, color=colors.grey, spaceBefore=8, spaceAfter=12),
    ]

    # Task 1 (ASCII-safe)
    story += [Paragraph("Task 1: Representation & Model", styles["H2"])]
    story += [Paragraph(
        "Ratings from MovieLens form a sparse user-item matrix. "
        "We factorize with Truncated SVD (k = 20) and persist three artifacts: "
        "U_all (user factors), Sigma (singular values), and V^T (item factors).",
        styles["Body"]
    )]
    story += [Paragraph(
        "For a new participant we estimate their vector in closed form from the ratings they give. "
        "Predicted ratings are computed as dot-products between the user vector and item factors.",
        styles["Body"]
    )]

    # artifacts table (ASCII shapes)
    t = Table(
        [
            ["Artifact", "Shape", "Purpose"],
            ["U_all", "n_users x 20", "Original users in latent space"],
            ["Sigma (vector)", "20", "Singular values for scaling"],
            ["V^T", "20 x n_items", "Item factors"],
        ],
        colWidths=[120, 120, None],
        hAlign="LEFT",
    )
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f0f0f0")),
        ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("BOTTOMPADDING", (0,0), (-1,0), 6),
    ]))
    story += [t, Spacer(1, 8)]

    # Task 2 (ASCII-safe)
    story += [Paragraph("Task 2: Guided Active Learning Strategy", styles["H2"])]
    story += [Paragraph(
        "Goal: learn a useful profile from a small number of ratings. "
        "For each query we show a brief explanation to help the participant decide.",
        styles["Body"]
    )]
    bullets = [
        "Confidence - how far the predicted rating is from neutral (3.0).",
        "Genre hint - dominant genre for a quick mental model of the taste dimension.",
        "Impact preview - 'Before' and 'After' Top-5 if the movie were rated 5.0.",
        "Information gain - change in the user vector from that hypothetical update.",
    ]
    # Use numbered list to avoid Unicode bullet glyphs
    story += [ListFlowable([ListItem(Paragraph(b, styles["Body"])) for b in bullets],
                           bulletType="1", leftIndent=14)]

    # Task 3 (ASCII-safe)
    story += [Paragraph("Task 3: Interface & Study Design", styles["H2"])]
    story += [Paragraph("Flow", styles["H3"])]
    flow = [
        "Consent -> Pre-survey -> Study (ratings) -> Post-survey -> Debrief.",
        "Two conditions: Guided (explanations) vs Control (no explanations).",
        "Assignment is randomized once per participant and stored in session.",
    ]
    story += [ListFlowable([ListItem(Paragraph(x, styles["Body"])) for x in flow], bulletType="1", leftIndent=14)]

    story += [Paragraph("Interface", styles["H3"])]
    story += [Paragraph(
        "The study screen shows the current Top-5, the projected Top-5 if the next movie is loved, and a rating form. "
        "A separate details page presents confidence, genre hint, impact preview, and information gain.",
        styles["Body"]
    )]

    story += [Paragraph("Logging", styles["H3"])]
    log_points = [
        "Anonymous participant id, condition, optional pilot flag.",
        "Timestamps and response time for rating submissions.",
        "Rating value, confidence, information gain.",
        "Top-5 lists (current and predicted) at decision time.",
    ]
    story += [ListFlowable([ListItem(Paragraph(x, styles["Body"])) for x in log_points],
                           bulletType="1", leftIndent=14)]

    # Ethics
    story += [Paragraph("Ethics & Data", styles["H2"])]
    ethics = [
        "Participants are anonymous; ids are random UUIDs stored in session.",
        "Only minimal data are kept for analysis; no free-text is collected.",
        "A Debrief page explains the study at the end.",
        "Data deletion is available on request.",
    ]
    story += [ListFlowable([ListItem(Paragraph(x, styles["Body"])) for x in ethics],
                           bulletType="1", leftIndent=14)]

    story += [Spacer(1, 10), Paragraph("Notes", styles["H3"])]
    story += [Paragraph(
        "SVD was chosen for simplicity and speed. The math is explicit in code, and the interface avoids jargon. "
        "Numbers shown to participants are rounded for readability.",
        styles["Body"]
    )]

    doc.build(story)
    buf.seek(0)
    resp = HttpResponse(buf, content_type="application/pdf")
    resp["Content-Disposition"] = 'attachment; filename="project4_documentation.pdf"'
    return resp

# ------------------------------------------------------------------------------------
# Consent / Pre / Post / Debrief
# ------------------------------------------------------------------------------------
def consent(request):
    pid = _get_pid(request); cond = _get_condition(request); pilot = _is_pilot(request)
    if request.method == 'POST':
        # No checkbox, so just log agreement automatically
        _log_event({
            'event': 'consent',
            'pid': pid,
            'condition': cond,
            'pilot': pilot,
            'agreed': 1
        })
        return redirect('project4:pre')
    return render(request, 'project4/consent.html')


def pre_survey(request):
    pid = _get_pid(request); cond = _get_condition(request); pilot = _is_pilot(request)
    if request.method == 'POST':
        _log_event({
            'event': 'pre', 'pid': pid, 'condition': cond, 'pilot': pilot,
            'trust': request.POST.get('trust'),
            'usefulness': request.POST.get('usefulness'),
            'expertise': request.POST.get('expertise'),
        })
        request.session['ratings'] = {}
        request.session['ratings_done'] = 0
        request.session['last_get_ts'] = None
        return redirect('project4:study')
    return render(request, 'project4/pre.html')

def post_survey(request):
    pid = _get_pid(request); cond = _get_condition(request); pilot = _is_pilot(request)
    if request.method == 'POST':
        _log_event({
            'event': 'post', 'pid': pid, 'condition': cond, 'pilot': pilot,
            'trust': request.POST.get('trust'),
            'usefulness': request.POST.get('usefulness'),
            'effort': request.POST.get('effort'),
        })
        return redirect('project4:debrief')
    return render(request, 'project4/post.html')

def debrief(request):
    pid = _get_pid(request)
    return render(request, 'project4/debrief.html', {'pid': pid})

# ------------------------------------------------------------------------------------
# Study (guided vs control) + logging + Details page
# ------------------------------------------------------------------------------------
def study(request):
    """
    Interactive cold-start study:
      - GUIDED: details are available on a separate page (confidence, genre, impact preview, info-gain).
      - CONTROL: same study flow but without the details link (template may hide it).
      - Logs ratings (+ timing, confidence, info-gain, top-5) to CSV.
    """
    pid  = _get_pid(request)
    cond = _get_condition(request)     # 'guided' | 'control'
    pilot = _is_pilot(request)

    # ratings dict in session
    if 'ratings' not in request.session:
        request.session['ratings'] = {}
    ratings = request.session['ratings']  # {movieId_str: float}

    # stop after N ratings
    if request.session.get('ratings_done', 0) >= STOP_AFTER_RATINGS:
        return redirect('project4:post')

    # build current profile & current recs (BEFORE any new rating comes in)
    if ratings:
        u_current = compute_new_user_profile({int(m): r for m, r in ratings.items()}, VT, movie_ids)
        current_recs = recommend(u_current, VT, movie_ids, movie_titles, top_n=5)
    else:
        u_current = None
        current_recs = []

    # ---------------------------
    # POST: user just submitted a rating
    # ---------------------------
    if request.method == 'POST':
        try:
            movie_id_raw = request.POST.get('movie_id', '').strip()
            rating_raw   = request.POST.get('rating', '').strip()
            if movie_id_raw == '' or rating_raw == '':
                raise ValueError("Missing movie_id or rating")

            movie_id = int(movie_id_raw)
            rating_val = float(rating_raw)
            if not (0.5 <= rating_val <= 5.0):
                raise ValueError(f"Out-of-range rating {rating_val}")
        except Exception:
            # On bad input, just refresh the page with a new prompt
            return redirect('project4:study')

        movie_title = (movie_titles.get(movie_id, f"Movie {movie_id}")
                       if isinstance(movie_titles, dict) else movie_titles[movie_id])

        # compute metrics for THIS movie_id based on the profile BEFORE adding this rating
        if u_current is not None and movie_id in MOVIE_INDEX:
            idx = MOVIE_INDEX[movie_id]
            pred_rating = float(u_current @ VT[:, idx])
        else:
            pred_rating = 3.0

        confidence = abs(pred_rating - 3.0) / 2.0 * 100.0
        confidence = max(0.0, min(100.0, confidence))  # clamp

        # hypothetical "Loved It" (5.0) preview
        hypo = {int(m): r for m, r in ratings.items()}
        hypo[movie_id] = 5.0
        u_after = compute_new_user_profile(hypo, VT, movie_ids)
        predicted_recs = recommend(u_after, VT, movie_ids, movie_titles, top_n=5)

        # info gain
        info_gain = float(np.linalg.norm(u_after - u_current)) if u_current is not None else float(np.linalg.norm(u_after))

        # response time (vs last GET time)
        last_get = request.session.get('last_get_ts')
        rt_ms = (_now_ms() - last_get) if last_get else None

        # log the event BEFORE updating ratings
        _log_event({
            'event': 'rating',
            'pid': pid,
            'condition': cond,
            'pilot': pilot,
            'movie_id': movie_id,
            'movie_title': movie_title,
            'rating': rating_val,
            'rt_ms': rt_ms,
            'confidence': round(confidence, 3),
            'info_gain': round(info_gain, 3),
            'top5_current': json.dumps(current_recs),
            'top5_predicted': json.dumps(predicted_recs),
        })

        # now update session with the user's rating
        ratings[str(movie_id)] = rating_val
        request.session['ratings'] = ratings
        request.session['ratings_done'] = request.session.get('ratings_done', 0) + 1

        # Support both buttons: if an older template still posts 'rate_and_analyze', send to details
        action = request.POST.get('action', 'rate')
        if action == 'rate_and_analyze':
            return redirect('project4:details')
        return redirect('project4:study')

    # ---------------------------
    # GET: render next prompt
    # ---------------------------
    # choose a movie to ask about (random pool; could exclude already-rated if desired)
    all_mids = list(MOVIE_INDEX.keys())
    next_mid = int(np.random.choice(all_mids))
    next_title = movie_titles.get(next_mid, f"Movie {next_mid}") if isinstance(movie_titles, dict) else movie_titles[next_mid]

    # compute metrics for the chosen next movie (for Details page)
    if u_current is not None and next_mid in MOVIE_INDEX:
        idx = MOVIE_INDEX[next_mid]
        pred_rating = float(u_current @ VT[:, idx])
    else:
        pred_rating = 3.0

    confidence = abs(pred_rating - 3.0) / 2.0 * 100.0
    confidence = max(0.0, min(100.0, confidence))

    genres_str = movie_genres.get(next_mid, "")
    key_genre  = genres_str.split('|')[0] if genres_str else "General"

    hypo = {int(m): r for m, r in ratings.items()}
    hypo[next_mid] = 5.0
    u_after = compute_new_user_profile(hypo, VT, movie_ids)
    predicted_recs = recommend(u_after, VT, movie_ids, movie_titles, top_n=5)

    info_gain = float(np.linalg.norm(u_after - u_current)) if u_current is not None else float(np.linalg.norm(u_after))

    # pairs for the impact preview cards (used on Details page)
    guidance_pairs = list(zip(current_recs, predicted_recs))

    # remember GET time for response-time computation
    request.session['last_get_ts'] = _now_ms()

    # Stash details for the separate Details page (JSON-serializable only)
    request.session['details'] = {
        'condition': cond,
        'current_recs': list(current_recs),
        'predicted_recs': list(predicted_recs),
        'guidance_pairs': [[a, b] for a, b in guidance_pairs],
        'next_movie_id': next_mid,
        'next_movie_title': next_title,
        'confidence': f"{confidence:.0f}",
        'key_genre': key_genre,
        'info_gain': f"{info_gain:.3f}",
    }

    # render study page (kept intentionally lean)
    return render(request, 'project4/study.html', {
        'condition': cond,
        'rated': [(movie_titles[int(m)] if isinstance(movie_titles, dict) else movie_titles[int(m)], r)
                  for m, r in ratings.items()],
        'current_recs': current_recs,
        'predicted_recs': predicted_recs,
        'next_movie_id': next_mid,
        'next_movie_title': next_title,
    })

def details(request):
    """
    Show guidance info (confidence, genre, impact preview) moved off the study page.
    Falls back to Study if metrics aren't available yet.
    """
    d = request.session.get('details')
    if not d:
        return redirect('project4:study')
    # Convert list-of-lists back to pairs for easy templating
    gpairs = d.get('guidance_pairs') or []
    d['guidance_pairs'] = [{'before': p[0], 'after': p[1]} for p in gpairs if isinstance(p, (list, tuple)) and len(p) == 2]
    return render(request, 'project4/details.html', d)


def restart_study(request):
    """Clears session data so the study restarts fresh."""
    request.session['ratings'] = {}
    request.session['ratings_done'] = 0
    request.session['last_get_ts'] = None
    request.session['details'] = None
    return redirect('project4:study')
