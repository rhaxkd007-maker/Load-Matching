# -*- coding: utf-8 -*-
import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)

# ---------- CSV 로더(구분자 자동) ----------
def _read_first(candidates, seps=(',', ';', '\t')):
    """후보 파일/구분자 조합을 순서대로 시도하여 첫 성공 CSV를 반환."""
    last_err = None
    for name in candidates:
        path = os.path.join(BASE_DIR, name)
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            continue
        for sep in seps:
            try:
                df = pd.read_csv(path, sep=sep, encoding='utf-8-sig', low_memory=False)
                if len(df) > 0:
                    print(f"[LOAD] {name} (sep={sep!r}) -> {len(df)} rows / {len(df.columns)} cols")
                    return df
            except Exception as e:
                last_err = e
    raise FileNotFoundError(f"CSV not found or empty. Tried {candidates} / seps={seps}. LastErr={last_err}")

# ---------- 좌표 표준화 ----------
def _ensure_display_coords(df: pd.DataFrame, prefer_pairs):
    """display_lat/lng 없으면 우선순위 컬럼쌍에서 만들어 준다."""
    if "display_lat" in df.columns and "display_lng" in df.columns:
        return df
    for a, b in prefer_pairs:
        if a in df.columns and b in df.columns:
            df["display_lat"] = df[a].astype(float)
            df["display_lng"] = df[b].astype(float)
            return df
    if {"lat", "lng"}.issubset(df.columns):
        df["display_lat"] = df["lat"].astype(float)
        df["display_lng"] = df["lng"].astype(float)
    return df

# ---------- 한국 5권역 라벨러 ----------
def _assign_region_kr(lat: float, lng: float, city: str = "") -> str:
    """
    한국 5권역(경기도/강원도/충청도/경상도/전라도) 라벨링.
    1) 도시명 힌트가 있으면 우선 사용
    2) 없으면 좌표 기반 대략 경계로 폴백
    """
    s = (city or "").strip()
    if "강원" in s: return "강원도"
    if any(x in s for x in ["경남", "경북", "부산", "대구", "울산"]): return "경상도"
    if any(x in s for x in ["전남", "전북", "광주"]): return "전라도"
    if any(x in s for x in ["충남", "충북", "대전", "세종"]): return "충청도"
    if any(x in s for x in ["경기", "서울", "인천"]): return "경기도"

    # 좌표 폴백(거친 경계지만 실무용으로 충분)
    # 동부 북측
    if lng >= 128.0 and lat >= 37.0:
        return "강원도"
    # 동부 남측
    if lng >= 128.0 and lat < 37.0:
        return "경상도"
    # 서부 남측
    if lng <= 127.0 and lat < 36.5:
        return "전라도"
    # 중부
    if 126.8 <= lng <= 128.0 and 36.0 <= lat <= 37.3:
        return "충청도"
    # 수도권/기타
    return "경기도"

def _make_region_col(df: pd.DataFrame, city_col: str | None = None) -> pd.DataFrame:
    """display_lat/lng을 사용해 region을 생성. city_col 있으면 힌트로 사용."""
    if "display_lat" not in df.columns or "display_lng" not in df.columns:
        return df
    if city_col and city_col in df.columns:
        df["region"] = df.apply(
            lambda r: _assign_region_kr(float(r["display_lat"]), float(r["display_lng"]), str(r.get(city_col, ""))),
            axis=1
        )
    else:
        df["region"] = df.apply(
            lambda r: _assign_region_kr(float(r["display_lat"]), float(r["display_lng"])),
            axis=1
        )
    return df

# ---------- 메인 ----------
def load_korean_data():
    # 1) 파일 로드 (kr 버전 우선)
    train = _read_first(["train_region_kr.csv", "train.csv"])
    drivers = _read_first(["driver_region_kr.csv", "driver.csv"])

    # 2) 좌표 표준화
    train = _ensure_display_coords(train, [("poi_lat", "poi_lng"), ("pickup_lat", "pickup_lng")])
    drivers = _ensure_display_coords(
        drivers,
        [("display_lat", "display_lng"),
         ("driver_lat", "driver_lng"),
         ("accept_gps_lat", "accept_gps_lng"),
         ("got_gps_lat", "got_gps_lng")]
    )

    # 3) 주문 id 없으면 임시 id 부여
    if not any(c in train.columns for c in ["pickup_id", "order_id", "id"]):
        train["id"] = train.index.astype(str)

    # 4) 5권역 라벨 생성 (주문/기사 모두)
    pickup_city_col = "from_city_name" if "from_city_name" in train.columns else ("pickup_city" if "pickup_city" in train.columns else None)
    train = _make_region_col(train, pickup_city_col)

    driver_city_col = "from_city_name" if "from_city_name" in drivers.columns else None
    drivers = _make_region_col(drivers, driver_city_col)

    # 5) 로그
    print("[INFO] train cols head:", list(train.columns)[:25])
    print("[INFO] drivers cols head:", list(drivers.columns)[:25])
    print("[INFO] region counts (orders):", dict(train["region"].value_counts()))
    print("[INFO] region counts (drivers):", dict(drivers["region"].value_counts()))

    return train, drivers
