import csv, os, time, uuid
from django.conf import settings

APP_DIR  = os.path.join(settings.BASE_DIR, 'project4')
DATA_DIR = os.path.join(APP_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

LOG_PATH = os.path.join(DATA_DIR, 'study_log.csv')

def get_pid(request):
    """Anonymous participant id (persisted in session)."""
    if 'pid' not in request.session:
        request.session['pid'] = str(uuid.uuid4())
    return request.session['pid']

def get_condition(request):
    """Between-subjects randomization: guided vs control."""
    if 'condition' not in request.session:
        import random
        request.session['condition'] = random.choice(['guided', 'control'])
    return request.session['condition']

def is_pilot(request):
    """Mark this run as pilot if ?pilot=1 is present once."""
    if 'pilot' not in request.session:
        request.session['pilot'] = 1 if request.GET.get('pilot') == '1' else 0
    return request.session['pilot']

def log_event(row: dict):
    """Append a row to CSV; create header if file is new."""
    new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if new:
            w.writeheader()
        w.writerow(row)

def now_ts():
    return int(time.time() * 1000)
