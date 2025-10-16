// ===== API =====
const api = {
  orders: "/api/orders",
  drivers: "/api/drivers",
  match: "/api/match",          // 서버가 radius_km 내 기사만 평가하도록 변경되어 있어야 함
  driversNear: "/api/drivers_near"
};

// ===== 유틸 =====
function detectIdKey(rows){
  for (const k of ['pickup_id','order_id','id']) if (rows.some(r => Object.hasOwn(r,k))) return k;
  return null;
}
function prefer(...vals){ for(const v of vals){ if(v!==undefined && v!==null && v===v) return v; } return null; }
function getOrderCoord(o){
  const lat = prefer(o.display_lat, o.poi_lat, o.pickup_lat, o.lat);
  const lng = prefer(o.display_lng, o.poi_lng, o.pickup_lng, o.lng);
  return (lat && lng) ? {lat: +lat, lng: +lng} : null;
}
async function j(url, opt){
  const u = url + (url.includes('?')?'&':'?') + '_ts=' + Date.now(); // 캐시 무력화
  const res = await fetch(u, opt || {cache:'no-store'});
  const text = await res.text();
  if(!res.ok){ console.error('[API ERROR]', url, res.status, text.slice(0,300)); throw new Error(url+' HTTP '+res.status); }
  return text ? JSON.parse(text) : [];
}
function log(msg){
  let el = document.getElementById('result');
  if (!el) { el = document.createElement('div'); el.id='result'; document.body.appendChild(el); }
  el.innerHTML = msg || '';
}
function ensureSelect(){
  let sel = document.getElementById('orderSelect') || document.querySelector('select#orderSelect') || document.querySelector('select');
  if (!sel) { sel = document.createElement('select'); sel.id='orderSelect'; document.querySelector('.controls')?.appendChild(sel); }
  return sel;
}

// ===== 지도 =====
const map = L.map('map').setView([36.5,127.9],7);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'&copy; OpenStreetMap contributors'}).addTo(map);
let orderMarkers=[], driverMarkers=[], routeLayer=null;

// ===== 상태 =====
let ORDER_ID_KEY=null, ORDERS=[], DRIVERS=[], SELECTABLE=[];

// ===== 데이터 로드 =====
async function loadData(){
  try{
    clearMap();
    ORDERS = await j(api.orders + '?limit=3000');
    DRIVERS = await j(api.drivers + '?limit=30000');
    ORDER_ID_KEY = detectIdKey(ORDERS);

    // --- 드롭다운: 권역 균등 샘플링 ---
    const sel = ensureSelect();
    sel.innerHTML = '<option value="">화물 선택…</option>';

    const withCoord = ORDERS.filter(o => getOrderCoord(o));
    const REGIONS = ['경기도','강원도','충청도','경상도','전라도'];
    const PER_REGION = 60;

    let bucket = [];
    for (const rg of REGIONS) bucket.push(...withCoord.filter(o => o.region === rg).slice(0, PER_REGION));
    if (bucket.length < 200) bucket.push(...withCoord.filter(o => !REGIONS.includes(o.region)).slice(0, 200 - bucket.length));
    SELECTABLE = bucket.length ? bucket : ORDERS.slice(0, 200);

    SELECTABLE.forEach((o, idx) => {
      let idVal = ORDER_ID_KEY ? o[ORDER_ID_KEY] : idx;
      if (idVal === undefined || idVal === null || idVal !== idVal) idVal = idx;
      sel.appendChild(new Option(`${o.region || '권역미상'} - ${String(idVal).slice(0,12)}…`, String(idVal)));
    });

    // --- 주문 마커(파란 점) ---
    const pairs = withCoord.map((o)=>({o,c:getOrderCoord(o)}));
    pairs.forEach(({o,c})=>{
      const m=L.circleMarker([c.lat,c.lng],{radius:5,color:'#1d4ed8'})
        .addTo(map).bindPopup(`<b>화물</b><br>${o.region||''}`);
      orderMarkers.push(m);
    });
    if (pairs.length) map.fitBounds(pairs.map(({c})=>[c.lat,c.lng]));

    log(`📦 화물 ${ORDERS.length} / 👷 기사 ${DRIVERS.length} · 드롭다운 ${Math.max(0, sel.options.length-1)}개` +
        ` · 강원 ${withCoord.filter(o=>o.region==='강원도').length} · 충청 ${withCoord.filter(o=>o.region==='충청도').length}`);
  }catch(e){ console.error(e); log('❌ 데이터 불러오기 실패'); }
}

function clearMap(){
  orderMarkers.forEach(m=>map.removeLayer(m)); orderMarkers=[];
  driverMarkers.forEach(m=>map.removeLayer(m)); driverMarkers=[];
  if(routeLayer){ map.removeLayer(routeLayer); routeLayer=null; }
}

// ===== 매칭(반경 내 기사만 평가) + TOP 색/라벨 =====
async function runMatch(){
  const sel = ensureSelect();
  const val = sel.value;
  if (!val) return log('화물을 선택하세요.');

  // 선택 화물
  let order = null;
  if (ORDER_ID_KEY) order = SELECTABLE.find(o => String(o[ORDER_ID_KEY]) === String(val));
  if (!order) order = SELECTABLE[Number(val)];
  if (!order) return log('선택한 화물을 찾지 못했습니다.');

  const oc = getOrderCoord(order);
  if (!oc) return log('선택 화물에 좌표가 없습니다.');

  // === 반경 설정(필요시 숫자만 바꾸면 정책 변경) ===
  const radiusKm = 60;      // 화물 인근 반경 (km)
  const nearLimit = 1200;   // 평가/표시 최대 기사 수

  // 1) 매칭: 반경 내 기사만 평가하도록 서버에 전달
  const body = {
    id_key: ORDER_ID_KEY || 'pickup_id',
    id_value: ORDER_ID_KEY ? order[ORDER_ID_KEY] : val,
    radius_km: radiusKm,
    near_limit: nearLimit
  };

  let data = [];
  try{
    data = await j(api.match, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });
  }catch(e){ console.error(e); return log('❌ 매칭 실패'); }
  if (!Array.isArray(data) || data.length===0) return log(`반경 ${radiusKm}km 내 추천 결과 없음`);

  // 2) 근처 기사도 동일 반경으로 조회(시야 = 평가대상과 동일)
  const near = await j(`${api.driversNear}?lat=${oc.lat}&lng=${oc.lng}&radius_km=${radiusKm}&limit=${nearLimit}`);

  // 기존 근처 마커 정리 후 빨간 점으로 그리기 + id→마커 매핑
  driverMarkers.forEach(m=>map.removeLayer(m)); driverMarkers=[];
  const markerById = new Map();
  const pickId = (o) => o?.delivery_user_id ?? o?.driver_id ?? o?.user_id ?? null;

  near.forEach(d=>{
    const m = L.circleMarker([d.display_lat, d.display_lng], {
      radius: 5,
      color: '#dc2626',
      fillColor: '#dc2626',
      fillOpacity: 1
    })
    .addTo(map)
    .bindPopup(`<b>기사</b><br>${d.region||''}<br><small>${d.delivery_user_id??''}</small>`);

    driverMarkers.push(m);
    const id = pickId(d);
    if (id != null) markerById.set(String(id), m);
  });

  // 3) 출발지 → TOP1 라인 (그대로)
  const best = data[0];
  if (best?.display_lat && best?.display_lng){
    if (routeLayer) map.removeLayer(routeLayer);
    routeLayer = L.polyline([[oc.lat,oc.lng],[best.display_lat,best.display_lng]], {color:'#16a34a'}).addTo(map);
    map.fitBounds(routeLayer.getBounds());
  }

  // 4) 근처 기사 중 TOP1~5만 강조(색/라벨)
  const rankColors = ['#16a34a',  // 1등: 초록
                      '#facc15',  // 2등: 노랑
                      '#f97316',  // 3등: 주황
                      '#a855f7',  // 4등: 보라
                      '#ef4444']; // 5등: 빨강(예비)

  function highlightRank(item, rankIdx){
    // id로 찾고, 없으면 좌표 근접으로 폴백
    const id = pickId(item);
    let m = (id != null) ? markerById.get(String(id)) : null;

    if (!m && item?.display_lat && item?.display_lng){
      const lat = +item.display_lat, lng = +item.display_lng;
      const eps = 1e-4; // ~10m 근접
      m = driverMarkers.find(mm => {
        const ll = mm.getLatLng();
        return Math.abs(ll.lat - lat) < eps && Math.abs(ll.lng - lng) < eps;
      });
    }
    if (!m) return;

    const color = rankColors[rankIdx] || '#ef4444';
    m.setStyle({ radius: 8, weight: 2, color: '#ffffff', fillColor: color, fillOpacity: 1 });

    const distKm = item.distance_km_calc ?? 0;
    m.unbindTooltip();
    m.bindTooltip(`${distKm.toFixed(2)} km`, {
      permanent: true, direction: 'top', offset: [0, -8], className: 'rank-label'
    });
    m.bringToFront();
  }

  data.slice(0, 5).forEach((d, i) => highlightRank(d, i));

  // 5) 결과 텍스트
  const html = data.map((d,i)=>`<div class="badge">TOP${i+1}</div> 거리: ${d.distance_km_calc?.toFixed?.(2) ?? '-'} km · 모델: ${(d.model_score ?? 0).toFixed(3)} · 총점: ${d.total_score?.toFixed?.(3)}`).join('<br>');
  log(html);
}

// ===== 이벤트 =====
document.getElementById('btnLoad').addEventListener('click', loadData);
document.getElementById('btnMatch').addEventListener('click', runMatch);
loadData();
