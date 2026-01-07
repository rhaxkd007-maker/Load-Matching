머신러닝을 통한 거리, 시간, 위치 기반 자동 매칭을 구현한 시스템

-Languages-
Python – 데이터 전처리, 피처엔지니어링, 추천/거리 계산, 모델 서빙(Flask)
JavaScript – 지도 UI(Leaflet), 비동기 API 연동, 시각화/인터랙션
HTML/CSS – 대시보드/컨트롤 패널

-Frameworks & Libraries-
Backend: Flask, Flask-CORS
ML/RecSys: XGBoost, NumPy, Pandas
Geo: Haversine(직접구현)
Frontend: Leaflet (OpenStreetMap), Fetch API

-Models-
XGBoost

-APIs (Flask)-
GET /api/orders – 화물 목록(표준 좌표 필드 보강)
GET /api/drivers – 기사 목록(좌표 필드 통일)
POST /api/match – 화물→상위 기사 추천
GET /api/drivers_near – 특정 좌표 반경 내 기사 서브셋(성능 최적화)
GET /api/health – 데이터/모델 상태 점검

-Frontend- 기능 (Leaflet UI)

화물 선택 드롭다운(권역 균형 샘플링)

화물/기사 마커 렌더링, 상위 1~3위 색상 구분(초록/노랑/주황)

TOP1 경로 폴리라인 및 거리 툴팁 표시

실시간 결과 패널(거리/모델점수/총점)

-Data-
CSV 기반 정형데이터(orders/drivers, 좌표/시간/지역/유형/랭크 등).
중국→한국 지명/좌표 변환 시 매핑 테이블 적용 및 좌표 노이즈 보정.

-Tools & Dev-
VS Code, Git/GitHub, Jupyter, Joblib

-배포/운영-
로컬 개발서버(Flask) + 정적 프론트엔드(Leaflet). 모델은 model_xgb.pkl 번들로 로드·갱신

https://youtu.be/1McsagsUZ2g   시연영상 링크
파란점: 화물 위치
초록점: 1순위 기사
노란점: 2순위 기사
주황점: 3순위 기사
빨간점: 근처의 다른 기사
