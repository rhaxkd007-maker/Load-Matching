# -*- coding: utf-8 -*-
"""중국 5대 도시 데이터를 한국 5도 좌표로 변환하는 스크립트
사용법(Windows PowerShell 예):
  python convert_cn_to_kr.py --train server/train.csv --driver server/driver.csv ^
         --out-train server/train_region_kr.csv --out-driver server/driver_region_kr.csv
"""
import pandas as pd
import hashlib
import argparse

REGIONS = {
  "경기도": [
    36.9,
    37.9,
    126.6,
    127.4
  ],
  "경상도": [
    35.0,
    36.2,
    128.0,
    129.2
  ],
  "전라도": [
    34.5,
    35.6,
    126.5,
    127.3
  ],
  "충청도": [
    36.0,
    36.9,
    126.8,
    127.7
  ],
  "강원도": [
    37.1,
    38.2,
    127.5,
    129.0
  ]
}
CN2KR = {
  "杭州市": "경기도",
  "重庆市": "경상도",
  "上海市": "전라도",
  "烟台市": "충청도",
  "吉林市": "강원도"
}

def det_uniform(key: str, a: float, b: float) -> float:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    u = int(h[:8], 16) / 16**8
    return a + (b - a) * u

def pick_point(region: str, unique_key: str):
    lat_min, lat_max, lon_min, lon_max = REGIONS[region]
    lat_pad = (lat_max - lat_min) * 0.15
    lon_pad = (lon_max - lon_min) * 0.15
    lat_a, lat_b = lat_min + lat_pad, lat_max - lat_pad
    lon_a, lon_b = lon_min + lon_pad, lon_max - lon_pad
    lat = det_uniform(unique_key + ":lat", lat_a, lat_b)
    lon = det_uniform(unique_key + ":lon", lon_a, lon_b)
    return lat, lon

def assign_region(city: str):
    if city in CN2KR:
        return CN2KR[city]
    buckets = list(REGIONS.keys())
    idx = int(hashlib.md5(str(city).encode("utf-8")).hexdigest()[:2], 16) % len(buckets)
    return buckets[idx]

def convert(train_path, driver_path, out_train, out_driver):
    train = pd.read_csv(train_path)
    driver = pd.read_csv(driver_path)

    # orders
    regions, lats, lngs = [], [], []
    for i, row in train.iterrows():
        city = str(row.get("pickup_city", ""))
        region = assign_region(city)
        key = str(row.get("pickup_id", i))
        lat, lon = pick_point(region, key)
        regions.append(region); lats.append(lat); lngs.append(lon)
    train["region"] = regions; train["display_lat"] = lats; train["display_lng"] = lngs
    train.to_csv(out_train, index=False)

    # drivers
    regions, lats, lngs = [], [], []
    for i, row in driver.iterrows():
        city = str(row.get("from_city_name", ""))
        region = assign_region(city)
        key = str(row.get("delivery_user_id", i))
        lat, lon = pick_point(region, key)
        regions.append(region); lats.append(lat); lngs.append(lon)
    driver["region"] = regions; driver["display_lat"] = lats; driver["display_lng"] = lngs
    driver.to_csv(out_driver, index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="원본 train.csv 경로")
    ap.add_argument("--driver", required=True, help="원본 driver.csv 경로")
    ap.add_argument("--out-train", default="train_region_kr.csv", help="출력 화물 CSV")
    ap.add_argument("--out-driver", default="driver_region_kr.csv", help="출력 기사 CSV")
    args = ap.parse_args()
    convert(args.train, args.driver, args.out_train, args.out_driver)
