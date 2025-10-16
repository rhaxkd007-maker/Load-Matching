# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Optional, Any, Dict, List

# -------------------- 공통 --------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def _pick(df, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns: return c
    return None

def _get_order_coord(row: pd.Series) -> Optional[tuple]:
    for a,b in [("display_lat","display_lng"),
               ("poi_lat","poi_lng"),
               ("pickup_lat","pickup_lng"),
               ("lat","lng")]:
        if a in row.index and b in row.index and pd.notna(row[a]) and pd.notna(row[b]):
            return float(row[a]), float(row[b])
    return None

def _stable_hash_num(x: Any, modulus: int = 10000) -> int:
    """라벨인코더가 없을 때를 위한 안정 해시 수치화(세션 불변)."""
    try:
        return int(abs(hash(str(x))) % modulus)
    except Exception:
        return 0

def _extract_hour_weekday(order: pd.Series) -> Dict[str, int]:
    """accept_time -> book_start_time -> expect_got_time -> got_time 순으로 시각 추출."""
    for k in ["accept_time", "book_start_time", "expect_got_time", "got_time"]:
        if k in order.index and pd.notna(order[k]):
            try:
                ts = pd.to_datetime(order[k])
                return {"hour": int(ts.hour), "weekday": int(ts.weekday())}
            except Exception:
                continue
    return {"hour": 0, "weekday": 0}

# -------------------- 예측 --------------------
def _predict_optional(bundle, X: pd.DataFrame) -> np.ndarray:
    """
    bundle이 없거나 오류면 0점, 있으면 Booster/Sklearn 모두 지원.
    예측값이 상수(pmin==pmax)면 0.0만 나오는 것을 막기 위해 아주 작은 지터를 섞어 정규화.
    """
    if not bundle or X is None or len(X) == 0:
        return np.zeros(len(X) if isinstance(X, pd.DataFrame) else 0)

    # feature_names가 있다면 순서 맞춰 한 번 더 재정렬(안전)
    fns = bundle.get("feature_names")
    if fns:
        X = pd.DataFrame({c: (X[c] if c in X.columns else 0.0) for c in fns})

    mdl = bundle.get("model")
    try:
        if bundle.get("type") == "booster":
            dmat = xgb.DMatrix(X.values, feature_names=list(X.columns))
            pred = mdl.predict(dmat)
        else:
            pred = mdl.predict(X)
    except Exception:
        return np.zeros(len(X))

    pred = np.asarray(pred).reshape(-1)

    # 상수 방지용 미세 노이즈
    if pred.size > 0 and float(np.max(pred)) == float(np.min(pred)):
        # 1e-12 수준의 매우 작은 차이를 만들어서 정규화가 0으로 죽지 않도록
        pred = pred + (np.arange(len(pred)) * 1e-12)

    pmin, pmax = float(np.min(pred)), float(np.max(pred))
    return (pred - pmin) / (pmax - pmin) if pmax > pmin else np.zeros_like(pred)

def _build_features_for_model(order: pd.Series,
                              cand_drivers: pd.DataFrame,
                              model_bundle) -> pd.DataFrame:
    """
    모델이 저장한 feature_names 순서로 주문×기사 피처를 생성.
    없으면 합리적 최소 피처셋으로 구성.
    """
    fns: List[str] = model_bundle.get("feature_names", []) if model_bundle else []
    lat_col = _pick(cand_drivers, ["display_lat","driver_lat","accept_gps_lat","got_gps_lat","lat"])
    lng_col = _pick(cand_drivers, ["display_lng","driver_lng","accept_gps_lng","got_gps_lng","lng"])

    o_lat, o_lng = _get_order_coord(order)
    dist = haversine(o_lat, o_lng,
                     cand_drivers[lat_col].astype(float).values,
                     cand_drivers[lng_col].astype(float).values)
    time_feat = _extract_hour_weekday(order)

    # 주문 원천 값
    pickup_city = order.get("from_city_name") or order.get("pickup_city")
    aoi_id      = order.get("aoi_id")
    typecode    = order.get("typecode")
    pickup_id   = order.get("pickup_id") if "pickup_id" in order.index else order.get("id")

    # 드라이버 랭크(없으면 0)
    rank_series = cand_drivers["rank"] if "rank" in cand_drivers.columns else 0.0

    cols: Dict[str, np.ndarray] = {}
    for c in (fns or []):
        if c == "pickup_lat":
            cols[c] = np.full(len(cand_drivers), o_lat, dtype=float)
        elif c == "pickup_lng":
            cols[c] = np.full(len(cand_drivers), o_lng, dtype=float)
        elif c == "driver_lat":
            cols[c] = cand_drivers[lat_col].astype(float).values
        elif c == "driver_lng":
            cols[c] = cand_drivers[lng_col].astype(float).values
        elif c == "distance_km":
            cols[c] = dist.astype(float)
        elif c == "rank":
            cols[c] = np.asarray(rank_series, dtype=float)
        elif c == "hour":
            cols[c] = np.full(len(cand_drivers), time_feat["hour"], dtype=float)
        elif c == "weekday":
            cols[c] = np.full(len(cand_drivers), time_feat["weekday"], dtype=float)
        elif c == "pickup_id_lnc":
            cols[c] = np.full(len(cand_drivers), _stable_hash_num(pickup_id), dtype=float)
        elif c == "pickup_city_lnc":
            cols[c] = np.full(len(cand_drivers), _stable_hash_num(pickup_city), dtype=float)
        elif c == "aoi_id_lnc":
            cols[c] = np.full(len(cand_drivers), _stable_hash_num(aoi_id), dtype=float)
        elif c == "typecode_lnc":
            cols[c] = np.full(len(cand_drivers), _stable_hash_num(typecode), dtype=float)
        else:
            cols[c] = np.zeros(len(cand_drivers), dtype=float)

    # feature_names가 비어있으면 최소 피처셋
    if not fns:
        cols = {
            "pickup_lat":  np.full(len(cand_drivers), o_lat, dtype=float),
            "pickup_lng":  np.full(len(cand_drivers), o_lng, dtype=float),
            "driver_lat":  cand_drivers[lat_col].astype(float).values,
            "driver_lng":  cand_drivers[lng_col].astype(float).values,
            "distance_km": dist.astype(float),
            "rank":        np.asarray(rank_series, dtype=float),
            "hour":        np.full(len(cand_drivers), time_feat["hour"], dtype=float),
            "weekday":     np.full(len(cand_drivers), time_feat["weekday"], dtype=float),
        }
        fns = list(cols.keys())

    X = pd.DataFrame({k: cols[k] for k in fns})
    # NaN/inf 방지
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X

# -------------------- 메인 추천 --------------------
def recommend_driver(req_ids: Any, orders: pd.DataFrame, drivers: pd.DataFrame, model_bundle=None):
    # 1) 주문 식별
    id_key, id_val = None, None
    if isinstance(req_ids, dict):
        id_key = req_ids.get("id_key")
        id_val = req_ids.get("id_value")
        if id_val is None and req_ids.get("pickup_id") is not None:
            id_key, id_val = "pickup_id", req_ids["pickup_id"]
    else:
        id_key, id_val = "pickup_id", req_ids

    order = None
    for col in [id_key, "pickup_id", "order_id", "id"]:
        if col and col in orders.columns:
            hit = orders[orders[col].astype(str) == str(id_val)]
            if len(hit):
                order = hit.iloc[0]; break
    if order is None:
        if len(orders) == 0: return []
        order = orders.iloc[0]

    # 2) 주문 좌표
    oc = _get_order_coord(order)
    if oc is None: return []
    o_lat, o_lng = oc

    # 3) 후보 기사 (좌표 있는 행만)
    df = drivers.copy()
    lat_col = _pick(df, ["display_lat","driver_lat","accept_gps_lat","got_gps_lat","lat"])
    lng_col = _pick(df, ["display_lng","driver_lng","accept_gps_lng","got_gps_lng","lng"])
    if not lat_col or not lng_col: return []
    df = df[df[lat_col].notna() & df[lng_col].notna()].copy()
    if len(df) == 0: return []

    # 4) 거리/베이스스코어 (지도 표기는 distance_km_calc, 모델은 distance_km 사용)
    df["distance_km_calc"] = haversine(o_lat, o_lng,
                                       df[lat_col].astype(float).values,
                                       df[lng_col].astype(float).values)
    base = 1.0 / (df["distance_km_calc"] + 1.0)

    # 5) 모델 피처 생성 + 예측
    X = _build_features_for_model(order, df, model_bundle)
    model_score = _predict_optional(model_bundle, X)
    df["model_score"] = model_score

    # 6) tie-breaker(동점 방지) & 총점
    id_col = _pick(df, ["delivery_user_id", "driver_id", "user_id"])
    if id_col:
        df = df.drop_duplicates(subset=[id_col])
    df = df.drop_duplicates(subset=[lat_col, lng_col])

    if "rank" in df.columns:
        r = df["rank"].astype(float)
        rank_bonus = (r.max() - r) / (r.max() - r.min() + 1e-9)
    else:
        rank_bonus = 0.0

    if id_col:
        tiny_noise = df[id_col].astype(str).apply(lambda s: (abs(hash(s)) % 997) / 997.0) * 1e-6
    else:
        tiny_noise = 0.0

    # 가중치: 거리 0.65, 모델 0.32, 랭크 0.03
    df["total_score"] = 0.65 * base + 0.32 * df["model_score"] + 0.03 * rank_bonus + tiny_noise

    # 7) 프론트 좌표 보강
    if "display_lat" not in df.columns: df["display_lat"] = df[lat_col].astype(float)
    if "display_lng" not in df.columns: df["display_lng"] = df[lng_col].astype(float)

    # 8) 최종 정렬
    df = df.sort_values(["total_score", "distance_km_calc"], ascending=[False, True])
    return df.head(3).to_dict(orient="records")
