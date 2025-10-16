# -*- coding: utf-8 -*-
import os, json, joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory, make_response
from flask_cors import CORS

from data_loader import load_korean_data
from match_logic import recommend_driver

app = Flask(__name__)
CORS(app)

# ----- 경로 설정 -----
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
FRONTEND_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "frontend"))
STATIC_DIR   = os.path.join(FRONTEND_DIR, "static")

# ----- 모델 로드(선택) -----
model_bundle = None
mpath = os.path.join(BASE_DIR, "model_xgb.pkl")
if os.path.exists(mpath):
    try:
        model_bundle = joblib.load(mpath)  # {"type","model","feature_names"}
        print("[INFO] 모델 로드:", model_bundle.get("type"))
    except Exception as e:
        print("[WARN] 모델 로드 실패 → 거리기반 모드:", e)

# ----- 데이터 로드 -----
train_df, driver_df = load_korean_data()

# ----- 공통 JSON 응답 -----
def _json(payload):
    body = json.dumps(payload, ensure_ascii=False)
    resp = make_response(body)
    resp.mimetype = "application/json"
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["Content-Length"] = str(len(body.encode("utf-8")))
    return resp

# ----- 좌표 유틸 -----
def _ensure_display_latlng(df, is_driver=False):
    """display_lat/display_lng 없으면 가능한 컬럼에서 보충"""
    if "display_lat" in df.columns and "display_lng" in df.columns:
        return df

    cand_pairs_orders = [("display_lat","display_lng"),
                         ("poi_lat","poi_lng"),
                         ("pickup_lat","pickup_lng"),
                         ("lat","lng")]
    cand_pairs_drivers = [("display_lat","display_lng"),
                          ("driver_lat","driver_lng"),
                          ("accept_gps_lat","accept_gps_lng"),
                          ("got_gps_lat","got_gps_lng"),
                          ("lat","lng")]
    pairs = cand_pairs_drivers if is_driver else cand_pairs_orders
    for a,b in pairs:
        if a in df.columns and b in df.columns:
            df = df.copy()
            df["display_lat"] = df[a]
            df["display_lng"] = df[b]
            return df
    return df  # 그대로 반환(없으면 이후 필터에서 제거)

def _get_row_coord(row: pd.Series):
    for a,b in [("display_lat","display_lng"),
                ("poi_lat","poi_lng"),
                ("pickup_lat","pickup_lng"),
                ("lat","lng")]:
        if a in row.index and b in row.index and pd.notna(row[a]) and pd.notna(row[b]):
            return float(row[a]), float(row[b])
    return None

def _haversine_np(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# ===================== API =====================

@app.route("/api/orders")
def api_orders():
    limit = int(request.args.get("limit", "3000"))
    out = _ensure_display_latlng(train_df.head(limit).copy(), is_driver=False)

    # id 컬럼 보장
    if not any(c in out.columns for c in ["pickup_id","order_id","id"]):
        out["id"] = out.index.astype(str)

    return _json(out.to_dict(orient="records"))

@app.route("/api/drivers")
def api_drivers():
    limit = int(request.args.get("limit", "30000"))
    out = _ensure_display_latlng(driver_df.head(limit).copy(), is_driver=True)
    return _json(out.to_dict(orient="records"))

@app.route("/api/drivers_near")
def api_drivers_near():
    """
    쿼리: lat, lng, radius_km(=60), limit(=2000)
    반환: 반경 내 기사(거리 오름차순)
    """
    try:
        lat = float(request.args.get("lat"))
        lng = float(request.args.get("lng"))
    except Exception:
        return _json([])

    radius_km = float(request.args.get("radius_km", 60))
    limit = int(request.args.get("limit", 2000))

    df = _ensure_display_latlng(driver_df.copy(), is_driver=True)
    if "display_lat" not in df.columns or "display_lng" not in df.columns:
        return _json([])

    df = df[df["display_lat"].notna() & df["display_lng"].notna()].copy()
    if len(df) == 0:
        return _json([])

    dist = _haversine_np(lat, lng, df["display_lat"].astype(float).values, df["display_lng"].astype(float).values)
    df["distance_km_calc"] = dist
    df = df.sort_values("distance_km_calc")
    df = df[df["distance_km_calc"] <= radius_km].head(limit)

    return _json(df.to_dict(orient="records"))

# ---- /api/match : 반경 내 기사만 평가 ----
@app.route("/api/match", methods=["POST"])
def api_match():
    """
    입력(JSON):
      - id_key, id_value : 주문 식별
      - radius_km(=60), near_limit(=2000)
    처리:
      1) 주문 좌표 구함
      2) 반경 내 기사 서브셋 추출
      3) recommend_driver(req, train_df, cand_df, model_bundle) 호출
    """
    req = request.get_json(force=True, silent=True) or {}
    radius_km = float(req.get("radius_km", 60))
    near_limit = int(req.get("near_limit", 2000))

    # 1) 주문 찾기
    id_key = req.get("id_key")
    id_val = req.get("id_value")
    order_row = None
    for col in [id_key, "pickup_id", "order_id", "id"]:
        if col and col in train_df.columns:
            hit = train_df[train_df[col].astype(str) == str(id_val)]
            if len(hit):
                order_row = hit.iloc[0]
                break
    if order_row is None:
        if len(train_df) == 0:
            return _json([])
        order_row = train_df.iloc[0]

    # 2) 주문 좌표
    oc = _get_row_coord(order_row)
    if oc is None:
        return _json([])

    # 3) 반경 내 기사만 추리기
    df = _ensure_display_latlng(driver_df.copy(), is_driver=True)
    if "display_lat" not in df.columns or "display_lng" not in df.columns:
        return _json([])

    df = df[df["display_lat"].notna() & df["display_lng"].notna()].copy()
    if len(df) == 0:
        return _json([])

    o_lat, o_lng = oc
    dist = _haversine_np(o_lat, o_lng, df["display_lat"].astype(float).values, df["display_lng"].astype(float).values)
    df["distance_km_calc"] = dist

    cand_df = df.sort_values("distance_km_calc")
    cand_df = cand_df[cand_df["distance_km_calc"] <= radius_km].head(near_limit)
    if len(cand_df) == 0:
        return _json([])

    # 4) 후보만 넘겨서 추천
    matched = recommend_driver(req, train_df, cand_df, model_bundle)
    return _json(matched)

@app.route("/api/health")
def api_health():
    try:
        overlap = None
        if model_bundle:
            fns = model_bundle.get("feature_names") or []
            possible = set(train_df.columns) | set(driver_df.columns) | {
                "display_lat","display_lng","driver_lat","driver_lng",
                "distance_km","rank","hour","weekday",
                "pickup_lat","pickup_lng",
                "pickup_id_lnc","pickup_city_lnc","aoi_id_lnc","typecode_lnc",
            }
            overlap = sum(1 for c in fns if c in possible)
        return _json({
            "orders_rows": int(len(train_df)),
            "drivers_rows": int(len(driver_df)),
            "orders_cols": list(train_df.columns)[:80],
            "drivers_cols": list(driver_df.columns)[:80],
            "model_loaded": bool(model_bundle),
            "model_feature_count": len(model_bundle.get("feature_names", [])) if model_bundle else 0,
            "model_feature_overlap_guess": overlap
        })
    except Exception as e:
        return _json({"ok": False, "error": str(e)})

# ===================== 정적 =====================

@app.route("/")
def root():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory(STATIC_DIR, path)

@app.route("/frontend/<path:path>")
def frontend_files(path):
    return send_from_directory(FRONTEND_DIR, path)

@app.route("/favicon.ico")
def favicon():
    return ("", 204)

if __name__ == "__main__":
    print("[PATH] FRONTEND_DIR:", FRONTEND_DIR)
    print("[PATH] STATIC_DIR  :", STATIC_DIR)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "3000")), debug=True)
