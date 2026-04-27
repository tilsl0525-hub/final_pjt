import time
import random
from pathlib import Path
from collections import Counter, deque

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from ultralytics import YOLO

# ==========================================
# 1. 웹 페이지 기본 설정
# ==========================================
st.set_page_config(
    page_title="Battery Vision Monitoring",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 파일이 pages 폴더에 있으므로, 한 단계 위(parent.parent)인 최상위 폴더를 Root 경로로 잡습니다.
ROOT_DIR = Path(__file__).resolve().parent.parent

AUTO_THRESHOLD = 0.70
REVIEW_LOW = 0.45
MAX_HISTORY = 150
DISPLAY_IMAGE_MAX_H = 260
AUTO_INTERVAL_SEC = 2.0
CHART_HEIGHT = 210
SENSOR_CHART_HEIGHT = 115
MANUAL_TABLE_HEIGHT = 110
TARGET_FPS = 1.0
TARGET_DEFECT_RATE = 5.0

# ==========================================
# 2. 공통 CSS 스타일
# ==========================================
st.markdown("""
<style>
:root {
    --hyundai-blue:#012d74;
    --hyundai-light-blue:#0056b3;
    --safe-green:#2e7d32;
    --normal-blue:#1976d2;
    --danger-red:#d32f2f;
    --warning-orange:#f57c00;
    --review-yellow:#fbc02d;
    --bg-color:#eef2f5;
    --border-color:#e5e7eb;
    --text-muted:#667085;
}
.stApp { background-color:var(--bg-color); }
.block-container { max-width:100%!important; padding:0.15rem 0.35rem 0.25rem 0.35rem!important; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    background:#fff!important; border:1px solid var(--border-color)!important;
    border-radius:10px!important; box-shadow:0 2px 8px rgba(0,0,0,0.05)!important; padding:12px 14px!important;
}
.main-header {
    background:linear-gradient(90deg,var(--hyundai-blue),var(--hyundai-light-blue));
    height:42px; border-radius:10px; color:white; margin-bottom:6px; display:flex; align-items:center; padding-left:18px;
}
.main-header h1 { color:white!important; margin:0; font-size:20px; }
.kpi-container { display:grid; grid-template-columns:repeat(8,1fr); gap:6px; margin-bottom:5px; }
.kpi-box { background:#fff; padding:6px 6px; border-radius:8px; text-align:center; border-top-width:4px solid var(--hyundai-blue); box-shadow:0 2px 8px rgba(0,0,0,0.05); }
.kpi-box.safe { border-top-color:var(--safe-green); }
.kpi-box.defect { border-top-color:var(--danger-red); }
.kpi-box.review { border-top-color:var(--warning-orange); }
.kpi-box.warning { border-top-color:var(--review-yellow); }
.kpi-title { font-size:10px; color:#555; font-weight:800; margin-bottom:2px; }
.kpi-value { font-size:17px; font-weight:900; color:var(--hyundai-blue); line-height:1.0; }
.kpi-sub { font-size:9px; color:var(--text-muted); margin-top:2px; }
.safe-text { color:var(--safe-green); }
.defect-text { color:var(--danger-red); }
.review-text { color:var(--warning-orange); }
.warning-text { color:var(--review-yellow); }
.progress-wrap { background:#d9e2ec; border-radius:999px; height: 9px; overflow:hidden; margin:1px 0 3px 0; }
.progress-bar { height:100%; background:linear-gradient(90deg,var(--normal-blue),var(--safe-green)); }
.section-title { font-size:13px; color:var(--hyundai-blue); font-weight:900; margin-bottom:5px; padding-bottom:4px; border-bottom:2px solid #f0f0f0; }
.result-badge { display:inline-block; padding:6px 10px; border-radius:8px; color:white; font-size:15px; font-weight:900; text-align:center; width:100%; box-sizing:border-box; }
.badge-normal { background:var(--safe-green); }
.badge-defect { background:var(--danger-red); animation:borderPulse 1.5s infinite; }
.badge-review { background:var(--warning-orange); }
.info-line { font-size:11px; margin:3px 0; color:#333; }
.info-line b { color:var(--hyundai-blue); }
.status-box { padding:6px; border-radius:8px; font-size:11px; font-weight:800; margin-bottom:5px; }
.status-ok { background:#ecfdf5; color:var(--safe-green); border:1px solid #bbf7d0; }
.status-warn { background:#fffbeb; color:#b45309; border:1px solid #fde68a; }
.status-danger { background:#fef2f2; color:var(--danger-red); border:1px solid #fecaca; }
.worker-log { height:105px; overflow-y:auto; background:#fff; border-radius:8px; border:1px solid var(--border-color); padding:5px; }
.event-row { padding:5px 7px; margin-bottom:4px; border-radius:8px; font-size:10.5px; border-left:5px solid #cbd5e1; background:#f8fafc; }
.event-defect { border-left-color:var(--danger-red); background:#fff1f2; font-weight:900; }
.event-review { border-left-color:var(--warning-orange); background:#fff7ed; font-weight:800; }
.event-normal { border-left-color:var(--safe-green); background:#f0fdf4; }
.sensor-card { background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:5px 7px; margin-bottom:5px; }
.sensor-title { font-size:10px; color:#475569; font-weight:800; }
.sensor-value { font-size:14px; color:var(--hyundai-blue); font-weight:900; }
@keyframes borderPulse { 50% { box-shadow:0 0 15px rgba(211,47,47,0.45); } }
header[data-testid="stHeader"] { display: none; }
.block-container { padding-top: 0rem !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. 데이터 로딩 (ZIP 대신 기존 폴더 직접 참조)
# ==========================================
@st.cache_resource
def prepare_dataset():
    image_dir = ROOT_DIR / "battery_defect_group_split"

    if not image_dir.exists():
        st.error(f"이미지 폴더를 찾을 수 없습니다: {image_dir}")
        st.stop()

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(image_dir.rglob(ext))

    image_paths = [p for p in image_paths if any(part in ["good", "not_good"] for part in p.parts)]
    return sorted(image_paths), image_dir.name

@st.cache_resource
def load_model():
    model_path = ROOT_DIR / "best.pt"

    if not model_path.exists():
        st.error(f"best.pt 모델 파일을 찾을 수 없습니다: {model_path}")
        st.stop()

    return YOLO(str(model_path)), model_path.name

image_paths, dataset_name = prepare_dataset()
model, model_name = load_model()
CLASS_NAMES = model.names if hasattr(model, "names") else {}

if not image_paths:
    st.error("good / not_good 이미지 데이터를 찾지 못했습니다. 폴더 구조를 확인해주세요.")
    st.stop()

# ==========================================
# 4. 상태 초기화 및 공통 함수
# ==========================================
def init_state():
    defaults = {
        "total_input": len(image_paths), "completed": 0, "normal": 0, "defect": 0, "review": 0,
        "records": [], "manual_queue": deque(maxlen=40), "logs": deque(maxlen=40),
        "class_counter": Counter(), "last_result": None, "last_image": None,
        "last_infer_ms": 0.0, "manual_actions": [], "sensor_history": deque(maxlen=60),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def pct(a, b): return (a / b * 100) if b else 0

def get_true_label(path: Path):
    parts = list(path.parts)
    if "not_good" in parts: return "not_good"
    if "good" in parts: return "good"
    return "unknown"

def normalize_class_name(name):
    name = str(name).lower()
    if name in ["good", "normal", "ok"]: return "good"
    if name in ["not_good", "bad", "defect", "ng", "abnormal"]: return "not_good"
    return name

def make_display_image(image_rgb, max_h=DISPLAY_IMAGE_MAX_H):
    h, w = image_rgb.shape[:2]
    if h <= max_h: return image_rgb
    ratio = max_h / h
    return cv2.resize(image_rgb, (int(w * ratio), max_h), interpolation=cv2.INTER_AREA)

def make_labeled_image(image_rgb, result, pred_label, confidence):
    img = image_rgb.copy()
    color = {"NORMAL": (46, 125, 50), "DEFECT": (211, 47, 47), "REVIEW": (245, 124, 0)}.get(result, (1, 45, 116))
    text = f"{result} | {pred_label} | {confidence:.2f}"
    cv2.rectangle(img, (8, 8), (img.shape[1] - 8, 48), color, -1)
    cv2.putText(img, text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return img

def make_sensor_data(result_type):
    if result_type == "DEFECT":
        voltage, temp, anomaly_score = np.random.normal(3.58, 0.045), np.random.normal(32.0, 1.2), min(1.0, np.random.normal(0.78, 0.12))
    elif result_type == "REVIEW":
        voltage, temp, anomaly_score = np.random.normal(3.65, 0.035), np.random.normal(29.8, 0.9), min(1.0, np.random.normal(0.52, 0.14))
    else:
        voltage, temp, anomaly_score = np.random.normal(3.70, 0.025), np.random.normal(28.0, 0.6), max(0.0, np.random.normal(0.18, 0.08))
    return {"voltage": round(float(voltage), 3), "temperature": round(float(temp), 1), "anomaly_score": round(float(max(0.0, min(1.0, anomaly_score))), 2)}

def classify_from_probs(result):
    probs = getattr(result, "probs", None)
    if probs is None: return "unknown", 0.0
    top1, conf = int(probs.top1), float(probs.top1conf)
    pred_name = CLASS_NAMES.get(top1, str(top1)) if isinstance(CLASS_NAMES, dict) else str(top1)
    return normalize_class_name(pred_name), conf

def decide_result(pred_label, confidence):
    if confidence < REVIEW_LOW: return "REVIEW", "low confidence"
    if pred_label == "good": return ("NORMAL", "classification good") if confidence >= AUTO_THRESHOLD else ("REVIEW", "good but low confidence")
    if pred_label == "not_good": return ("DEFECT", "classification not_good") if confidence >= AUTO_THRESHOLD else ("REVIEW", "not_good but low confidence")
    return "REVIEW", "unknown class"

def inspect_one(path: Path):
    image_bgr = cv2.imread(str(path))
    if image_bgr is None: return None

    start = time.perf_counter()
    result = model.predict(image_bgr, verbose=False)[0]
    infer_ms = (time.perf_counter() - start) * 1000

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pred_label, confidence = classify_from_probs(result)
    final_result, reason = decide_result(pred_label, confidence)

    display_rgb = make_display_image(make_labeled_image(image_rgb, final_result, pred_label, confidence))
    true_label, cell_id, now = get_true_label(path), path.stem.replace("RGB_cell_", "CELL_"), time.strftime("%H:%M:%S")
    sensor = make_sensor_data(final_result)

    record = {
        "time": now, "cell_id": cell_id, "result": final_result, "pred_label": pred_label,
        "true_label": true_label, "confidence": confidence, "reason": reason, "infer_ms": infer_ms,
        "voltage": sensor["voltage"], "temperature": sensor["temperature"], "anomaly_score": sensor["anomaly_score"], "file": str(path),
    }

    st.session_state.completed += 1
    if final_result == "NORMAL": st.session_state.normal += 1
    elif final_result == "DEFECT": st.session_state.defect += 1
    else: 
        st.session_state.review += 1
        st.session_state.manual_queue.appendleft(record)

    st.session_state.class_counter[pred_label] += 1
    st.session_state.records.append(record)
    if len(st.session_state.records) > MAX_HISTORY: st.session_state.records = st.session_state.records[-MAX_HISTORY:]
    st.session_state.sensor_history.append({"time": now, "voltage": sensor["voltage"], "temperature": sensor["temperature"], "anomaly_score": sensor["anomaly_score"]})

    log_type = final_result
    if final_result == "DEFECT": log_text = f"[{now}] DEFECT | {cell_id} | pred={pred_label} | conf={confidence:.2f}"
    elif final_result == "REVIEW": log_text = f"[{now}] REVIEW | {cell_id} | {reason} | conf={confidence:.2f}"
    else: log_text = f"[{now}] NORMAL | {cell_id} | conf={confidence:.2f} | {infer_ms:.1f}ms"

    st.session_state.logs.appendleft({"type": log_type, "text": log_text})
    st.session_state.last_result = record
    st.session_state.last_image = display_rgb
    st.session_state.last_infer_ms = infer_ms

    return record

def run_next(n=1):
    for _ in range(n): inspect_one(random.choice(image_paths))

def classify_fps_status(fps):
    if fps >= TARGET_FPS: return "OK", "status-ok", "처리 속도 정상"
    if fps >= TARGET_FPS * 0.8: return "WARN", "status-warn", "목표 FPS 근접. 병목 가능성 확인 필요"
    return "DANGER", "status-danger", "목표 FPS 미달. 모델/HW 최적화 필요"

def render_worker_logs():
    rows = []
    for log in list(st.session_state.logs)[:10]:
        cls = {"DEFECT": "event-row event-defect", "REVIEW": "event-row event-review", "NORMAL": "event-row event-normal"}.get(log["type"], "event-row")
        rows.append(f"<div class='{cls}'>{log['text']}</div>")
    if not rows: rows.append("<div class='event-row'>이벤트 없음</div>")
    st.markdown(f"<div class='worker-log'>{''.join(rows)}</div>", unsafe_allow_html=True)

# ==========================================
# 5. 화면 렌더링 (🚀 잔상 방지를 위한 전체 컨테이너 적용)
# ==========================================
run_next(1)

# 잔상이 생기지 않도록 전체 UI를 하나의 empty 컨테이너 안에서 그림
ui_container = st.empty()

with ui_container.container():
    st.markdown("""<div class="main-header"><h1>배터리 비전 검사 모니터링</h1></div>""", unsafe_allow_html=True)

    completed = st.session_state.completed
    total_input = st.session_state.total_input

    normal_rate, defect_rate, review_rate, progress_rate = pct(st.session_state.normal, completed), pct(st.session_state.defect, completed), pct(st.session_state.review, completed), pct(completed, total_input)

    records_df = pd.DataFrame(st.session_state.records)
    avg_ms = records_df["infer_ms"].mean() if not records_df.empty else 0
    fps, uph = (1000 / avg_ms if avg_ms else 0), (1000 / avg_ms if avg_ms else 0) * 3600
    fps_status, fps_status_class, fps_message = classify_fps_status(fps)

    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-box"><div class="kpi-title">전체 셀 수</div><div class="kpi-value">{total_input}</div></div>
        <div class="kpi-box"><div class="kpi-title">검사 완료</div><div class="kpi-value">{completed}</div><div class="kpi-sub">{progress_rate:.1f}% 진행</div></div>
        <div class="kpi-box safe"><div class="kpi-title">정상 판정</div><div class="kpi-value safe-text">{st.session_state.normal}</div><div class="kpi-sub">{normal_rate:.1f}%</div></div>
        <div class="kpi-box defect"><div class="kpi-title">불량 판정</div><div class="kpi-value defect-text">{st.session_state.defect}</div><div class="kpi-sub">{defect_rate:.1f}%</div></div>
        <div class="kpi-box review"><div class="kpi-title">수동 검수</div><div class="kpi-value review-text">{st.session_state.review}</div><div class="kpi-sub">{review_rate:.1f}%</div></div>
        <div class="kpi-box {'defect' if defect_rate > TARGET_DEFECT_RATE else 'safe'}"><div class="kpi-title">불량률 기준</div><div class="kpi-value {'defect-text' if defect_rate > TARGET_DEFECT_RATE else 'safe-text'}">{defect_rate:.1f}%</div><div class="kpi-sub">목표 {TARGET_DEFECT_RATE:.1f}% 이하</div></div>
        <div class="kpi-box {'defect' if fps_status == 'DANGER' else 'warning' if fps_status == 'WARN' else 'safe'}"><div class="kpi-title">처리 FPS</div><div class="kpi-value {'defect-text' if fps_status == 'DANGER' else 'warning-text' if fps_status == 'WARN' else 'safe-text'}">{fps:.2f}</div><div class="kpi-sub">목표 {TARGET_FPS:.1f} FPS</div></div>
        <div class="kpi-box"><div class="kpi-title">예상 처리량</div><div class="kpi-value">{uph:.0f}</div><div class="kpi-sub">UPH</div></div>
    </div>
    <div class="progress-wrap"><div class="progress-bar" style="width:{min(progress_rate, 100):.2f}%"></div></div>
    <div class="kpi-sub" style="margin-bottom:10px;">검사 진행률: {completed} / {total_input} cells · {progress_rate:.1f}%</div>
    """, unsafe_allow_html=True)

    main_col, side_col = st.columns([5.8, 4.2], gap="small")
    last = st.session_state.last_result or {}

    with main_col:
        with st.container(border=True):
            st.markdown("<div class='section-title'>실시간 분류 화면</div>", unsafe_allow_html=True)
            img_col, info_col = st.columns([3.2, 6.8], gap="small")
            with img_col:
                c1, c2, c3 = st.columns([1, 2, 1])
                with c2: st.image(st.session_state.last_image)
            with info_col:
                result = last.get("result", "-")
                badge_class = {"NORMAL": "badge-normal", "DEFECT": "badge-defect", "REVIEW": "badge-review"}.get(result, "badge-normal")
                st.markdown(f"<div class='result-badge {badge_class}'>{result}</div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='info-line'><b>셀 ID</b> : {last.get('cell_id', '-')}</div>
                <div class='info-line'><b>검사 시간</b> : {last.get('time', '-')}</div>
                <div class='info-line'><b>신뢰도</b> : {last.get('confidence', 0):.2f}</div>
                <div class='info-line'><b>추론 시간</b> : {last.get('infer_ms', 0):.1f} ms</div>
                <div class='info-line'><b>판정 기준</b> : 검수 {REVIEW_LOW:.2f} ~ 자동판정 {AUTO_THRESHOLD:.2f}</div>
                """, unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="small")
        with c1:
            with st.container(border=True):
                st.markdown("<div class='section-title'>정상 / 불량 / 검수 비율</div>", unsafe_allow_html=True)
                donut_df = pd.DataFrame({"type": ["Normal", "Defect", "Review"], "count": [st.session_state.normal, st.session_state.defect, st.session_state.review]})
                fig = px.pie(donut_df, values="count", names="type", hole=0.55, color="type", color_discrete_map={"Normal": "#2e7d32", "Defect": "#d32f2f", "Review": "#f57c00"})
                fig.update_layout(height=CHART_HEIGHT, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with c2:
            with st.container(border=True):
                st.markdown("<div class='section-title'>클래스별 예측 현황</div>", unsafe_allow_html=True)
                class_df = pd.DataFrame([{"class": "good", "count": st.session_state.class_counter.get("good", 0)}, {"class": "not_good", "count": st.session_state.class_counter.get("not_good", 0)}, {"class": "unknown", "count": st.session_state.class_counter.get("unknown", 0)}])
                fig = px.bar(class_df, x="count", y="class", orientation="h", text="count", color="class", color_discrete_map={"good": "#2e7d32", "not_good": "#d32f2f", "unknown": "#f57c00"})
                fig.update_layout(height=210, margin=dict(l=0, r=0, t=0, b=0), yaxis=dict(autorange="reversed"), xaxis_title=None, yaxis_title=None, showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        c3, c4 = st.columns(2, gap="small")
        with c3:
            with st.container(border=True):
                st.markdown("<div class='section-title'>불량 / 검수 발생 추세</div>", unsafe_allow_html=True)
                if records_df.empty: st.info("추세 데이터 없음")
                else:
                    trend_df = records_df.copy()
                    trend_df["seq"], trend_df["abnormal"] = range(1, len(trend_df) + 1), trend_df["result"].isin(["DEFECT", "REVIEW"]).astype(int)
                    trend_df["rolling_abnormal_rate"] = trend_df["abnormal"].rolling(20, min_periods=1).mean() * 100
                    fig = px.line(trend_df, x="seq", y="rolling_abnormal_rate")
                    fig.add_hline(y=TARGET_DEFECT_RATE, line_dash="dash", annotation_text="Target")
                    fig.update_layout(height=210, margin=dict(l=0, r=0, t=0, b=0), xaxis_title="검사 순번", yaxis_title="최근 이상률(%)")
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with c4:
            with st.container(border=True):
                st.markdown("<div class='section-title'>Confidence 분포</div>", unsafe_allow_html=True)
                if records_df.empty: st.info("분포 데이터 없음")
                else:
                    fig = px.histogram(records_df, x="confidence", nbins=12, color="result", color_discrete_map={"NORMAL": "#2e7d32", "DEFECT": "#d32f2f", "REVIEW": "#f57c00"})
                    fig.add_vline(x=REVIEW_LOW, line_dash="dash")
                    fig.add_vline(x=AUTO_THRESHOLD, line_dash="dash")
                    fig.update_layout(height=210, margin=dict(l=0, r=0, t=0, b=0), xaxis_title="Confidence", yaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with side_col:
        with st.container(border=True):
            st.markdown("<div class='section-title'>사람 검수 대기 셀</div>", unsafe_allow_html=True)
            manual_df = pd.DataFrame(list(st.session_state.manual_queue))
            if manual_df.empty: st.info("현재 수동 검수 대기 항목 없음")
            else:
                view = manual_df[["time", "cell_id", "pred_label", "confidence", "reason"]].head(8).copy()
                view["confidence"] = view["confidence"].map(lambda x: f"{x:.2f}")
                view = view.rename(columns={"time": "시간", "cell_id": "셀 ID", "pred_label": "예측", "confidence": "신뢰도", "reason": "사유"})
                selected_idx = st.selectbox("검수 대상 선택", options=list(range(len(view))), format_func=lambda i: f"{view.iloc[i]['시간']} | {view.iloc[i]['셀 ID']} | {view.iloc[i]['예측']}", label_visibility="collapsed")
                selected_record = list(st.session_state.manual_queue)[selected_idx]
                st.dataframe(view, hide_index=True, use_container_width=True, height=MANUAL_TABLE_HEIGHT)
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("정상 처리", use_container_width=True):
                        selected_record["manual_label"], selected_record["manual_time"] = "good", time.strftime("%H:%M:%S")
                        st.session_state.manual_actions.append(selected_record)
                        st.success(f"{selected_record['cell_id']} 정상 처리 완료")
                with b2:
                    if st.button("불량 확정", use_container_width=True):
                        selected_record["manual_label"], selected_record["manual_time"] = "not_good", time.strftime("%H:%M:%S")
                        st.session_state.manual_actions.append(selected_record)
                        st.error(f"{selected_record['cell_id']} 불량 확정 완료")

        with st.container(border=True):
            st.markdown("<div class='section-title'>전압 / 온도 / 내부 이상치</div>", unsafe_allow_html=True)
            anomaly_score = last.get("anomaly_score", 0)
            anomaly_class, anomaly_text = ("status-danger", "내부 데이터 이상 가능성 높음") if anomaly_score >= 0.70 else (("status-warn", "내부 데이터 관찰 필요") if anomaly_score >= 0.45 else ("status-ok", "내부 데이터 정상"))
            st.markdown(f"<div class='status-box {anomaly_class}'>{anomaly_text}</div>", unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            with s1: st.markdown(f"""<div class="sensor-card"><div class="sensor-title">전압</div><div class="sensor-value">{last.get('voltage', 0):.3f}V</div></div>""", unsafe_allow_html=True)
            with s2: st.markdown(f"""<div class="sensor-card"><div class="sensor-title">온도</div><div class="sensor-value">{last.get('temperature', 0):.1f}℃</div></div>""", unsafe_allow_html=True)
            with s3: st.markdown(f"""<div class="sensor-card"><div class="sensor-title">Anomaly</div><div class="sensor-value">{last.get('anomaly_score', 0):.2f}</div></div>""", unsafe_allow_html=True)
            sensor_df = pd.DataFrame(list(st.session_state.sensor_history))
            if not sensor_df.empty:
                sensor_df["seq"] = range(1, len(sensor_df) + 1)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sensor_df["seq"], y=sensor_df["voltage"], mode="lines", name="Voltage"))
                fig.add_trace(go.Scatter(x=sensor_df["seq"], y=sensor_df["temperature"] / 10, mode="lines", name="Temp/10"))
                fig.update_layout(height=SENSOR_CHART_HEIGHT, margin=dict(l=0, r=0, t=0, b=0), xaxis_title=None, yaxis_title=None, legend=dict(orientation="h", y=-0.25))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with st.container(border=True):
            st.markdown("<div class='section-title'>알람 / 이벤트 로그</div>", unsafe_allow_html=True)
            render_worker_logs()

        with st.container(border=True):
            st.markdown("<div class='section-title'>시스템 상태 / 병목 모니터링</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='status-box {fps_status_class}'>{fps_message}</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='info-line'><b>평균 추론시간</b> : {avg_ms:.1f} ms</div>
            <div class='info-line'><b>처리 FPS</b> : {fps:.2f}</div>
            <div class='info-line'><b>목표 FPS</b> : {TARGET_FPS:.2f}</div>
            <div class='info-line'><b>모델 파일</b> : {model_name}</div>
            <div class='info-line'><b>클래스</b> : good / not_good</div>
            """, unsafe_allow_html=True)

# 다음 스텝 실행을 위한 재귀 호출
time.sleep(AUTO_INTERVAL_SEC)
st.rerun()