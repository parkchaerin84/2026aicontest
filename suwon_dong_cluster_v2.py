# -*- coding: utf-8 -*-
"""
수원시 교통약자 이동유형 군집분석 v2 (행정동 단위, 35개 분석단위)
 입력
  - 경기도 등록장애인 집계현황(읍면동별·유형별·정도별, 2025.07)      : 장애유형 구성, 중증비율, 장애인 수
  - 국토교통부 전국 버스정류장 위치정보(2025.10.31)                     : 동별 정류장 수·밀도 (행정동 경계와 공간결합)
  - 수원시정연구원 정책Brief 제39호(2023.06)                            : 동별 고령화율·독거노인비율
  - 행정동 경계: github.com/vuski/admdongkor (ver20250101)
 주의: 장애인 CSV에서 장안구 8개 동은 '정자1동' 한 줄로 합산, 망포1·2동은 별도 행이 없음
       → 장안북부 8개 동을 1개 단위로, 망포1·2동은 영통3동에 포함해 분석(아래 MERGE 참고)
"""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.geometry import shape, Point
from shapely.ops import unary_union, transform
from shapely.strtree import STRtree
from pyproj import Transformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

UP, OUT = "/mnt/user-data/uploads/", "/mnt/user-data/outputs/"
GEO = "/tmp/hjd.geojson"   # curl -sL -o /tmp/hjd.geojson https://raw.githubusercontent.com/vuski/admdongkor/master/ver20250101/HangJeongDong_ver20250101.geojson
plt.rcParams["font.family"] = "Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"] = False

# ---------- 0. 단위 병합 규칙 ----------
N8 = ["정자1동","정자2동","정자3동","송죽동","연무동","영화동","조원1동","조원2동"]
MERGE = {d: "장안북부8동" for d in N8}
MERGE.update({"망포1동": "영통3동", "망포2동": "영통3동"})
unit_of = lambda d: MERGE.get(d, d)

# ---------- 1. 행정동 경계 (수원 44개) ----------
g = json.load(open(GEO, encoding="utf-8"))
to5179 = Transformer.from_crs(4326, 5179, always_xy=True).transform
polys, gu_of = {}, {}
for f in g["features"]:
    nm = f["properties"]["adm_nm"]
    if "수원시" not in nm: continue
    gu, dong = nm.split()[-2][-3:], nm.split()[-1]          # '수원시장안구' -> '장안구'
    polys[dong] = transform(to5179, shape(f["geometry"])); gu_of[dong] = gu
assert len(polys) == 44
unit_geom, unit_gu = {}, {}
for d, p in polys.items():
    unit_geom.setdefault(unit_of(d), []).append(p); unit_gu[unit_of(d)] = gu_of[d]
unit_geom = {u: unary_union(v) for u, v in unit_geom.items()}
area = {u: p.area / 1e6 for u, p in unit_geom.items()}   # ㎢

# ---------- 2. 정류장 -> 동 (공간결합) ----------
bs = pd.read_csv(UP + "국토교통부_전국_버스정류장_위치정보_20251031.csv", encoding="cp949", dtype=str)
bs = bs[bs["도시명"].str.contains("수원", na=False)].copy()
bs[["위도", "경도"]] = bs[["위도", "경도"]].astype(float)
names, geoms = list(unit_geom), list(unit_geom.values())
tree = STRtree(geoms)
cnt, outside = dict.fromkeys(names, 0), 0
for lon, lat in zip(bs["경도"], bs["위도"]):
    x, y = to5179(lon, lat); pt = Point(x, y)
    hit = tree.query(pt, predicate="within")
    if len(hit): cnt[names[hit[0]]] += 1
    else:
        i = tree.nearest(pt)                                   # 경계 밖(오차)은 가장 가까운 동으로
        if pt.distance(geoms[i]) < 300: cnt[names[i]] += 1
        outside += 1
print(f"정류장 {len(bs)}개 중 경계 밖 {outside}개(300m 이내 최근접 동에 배정), 배정 합계 {sum(cnt.values())}")

# ---------- 3. 등록장애인 ----------
d = pd.read_csv(UP + "경기도내등록장애인집계현황_읍면동별_유형별_정도별_.csv", encoding="cp949")
d = d[d["읍면동명"] != "광교동"].copy()                         # 1명짜리 행 제외
d["unit"] = d["읍면동명"].map(unit_of)
for c in ["등록장애인수(합계)", "심한장애인수(합계)"]: d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
pv = d.pivot_table(index="unit", columns="장애유형", values="등록장애인수(합계)", aggfunc="sum").fillna(0)
tot = pv.sum(axis=1); sev = d.groupby("unit")["심한장애인수(합계)"].sum()
grp = {"이동제약": ["지체", "뇌병변"], "감각": ["시각", "청각", "언어"], "인지정신": ["지적", "자폐성", "정신", "뇌전증"]}

# ---------- 4. 고령화율·독거노인비율 (PDF, 2023.06) ----------
raw = """송죽동 16.9 24.9|연무동 24.0 27.1|영화동 23.3 27.8|율천동 11.4 10.3|정자1동 13.9 31.8|정자2동 14.1 28.6|정자3동 8.8 21.1|조원1동 18.1 29.2|조원2동 16.4 47.3|파장동 19.1 29.0|
곡선동 7.9 7.3|구운동 16.8 23.2|권선1동 12.7 10.5|권선2동 10.1 20.1|금곡동 11.9 24.2|서둔동 16.5 21.7|세류1동 16.4 23.8|세류2동 18.3 20.3|세류3동 19.9 24.0|입북동 12.7 20.5|평동 14.9 26.4|호매실동 10.0 25.9|
고등동 15.9 23.1|매교동 10.2 15.9|매산동 19.5 17.1|우만1동 19.7 26.3|우만2동 12.7 13.7|인계동 12.6 10.0|지동 27.5 34.3|행궁동 26.4 26.6|화서1동 19.1 31.7|화서2동 13.0 24.8|
광교1동 6.2 4.7|광교2동 11.0 22.2|망포1동 7.4 15.9|망포2동 5.2 13.0|매탄1동 13.8 19.9|매탄2동 15.6 22.4|매탄3동 9.5 10.4|매탄4동 13.6 19.2|영통1동 8.6 19.4|영통2동 5.7 5.5|영통3동 7.5 8.1|원천동 9.5 11.6"""
age = pd.DataFrame([r.split() for r in raw.replace("\n", "").split("|")], columns=["동", "고령화율", "독거노인비율"])
age[["고령화율", "독거노인비율"]] = age[["고령화율", "독거노인비율"]].astype(float)
age["unit"] = age["동"].map(unit_of); age = age.groupby("unit")[["고령화율", "독거노인비율"]].mean()  # 병합단위는 단순평균

# ---------- 5. 특성표 ----------
df = pd.DataFrame(index=sorted(set(tot.index) & set(names)))
df["구"] = [unit_gu[u] for u in df.index]
df["장애인수"] = tot; df["정류장수"] = pd.Series(cnt); df["면적㎢"] = pd.Series(area)
df["정류장밀도(개/㎢)"] = df["정류장수"] / df["면적㎢"]
df["장애인/정류장(명)"] = df["장애인수"] / df["정류장수"].clip(lower=1)
df["중증비율(%)"] = sev / tot * 100
for k_, c_ in grp.items(): df[f"{k_}(%)"] = pv[c_].sum(axis=1) / tot * 100
df = df.join(age)
feats = ["정류장밀도(개/㎢)", "장애인/정류장(명)", "중증비율(%)", "이동제약(%)", "감각(%)", "인지정신(%)", "고령화율", "독거노인비율"]
Xr = df[feats].copy(); Xr["장애인/정류장(명)"] = np.log(Xr["장애인/정류장(명)"])   # 치우침 완화
X = StandardScaler().fit_transform(Xr)

# ---------- 6. 군집 ----------
sil = {k: silhouette_score(X, KMeans(k, n_init=50, random_state=0).fit_predict(X)) for k in range(2, 7)}
k = max([3, 4, 5], key=lambda i: sil[i])               # 해석 가능한 3~5개 중 실루엣 최대
lab = KMeans(k, n_init=50, random_state=0).fit_predict(X)
prof = df.assign(c=lab).groupby("c")[feats].mean()
z = (prof - df[feats].mean()) / df[feats].std()
order = z[["고령화율", "독거노인비율"]].sum(axis=1).sort_values(ascending=False).index   # 고령·독거 높은 순으로 번호
rm = {o: i for i, o in enumerate(order)}; df["군집"] = [rm[c] for c in lab]
prof = df.groupby("군집")[feats].mean(); z = (prof - df[feats].mean()) / df[feats].std()
short = {"정류장밀도(개/㎢)": "정류장밀도", "장애인/정류장(명)": "정류장당 장애인", "중증비율(%)": "중증", "이동제약(%)": "이동제약형",
         "감각(%)": "감각형", "인지정신(%)": "인지·정신형", "고령화율": "고령화", "독거노인비율": "독거노인"}
def nm(c):
    top = z.loc[c].reindex(z.loc[c].abs().sort_values(ascending=False).index)[:2]
    return f"C{c+1}: " + "·".join(f"{short[i]}{'↑' if v > 0 else '↓'}" for i, v in top.items())
cn = {c: nm(c) for c in prof.index}; df["군집명"] = df["군집"].map(cn)
df.round(2).to_csv(OUT + "suwon_dong_cluster_v2_table.csv", encoding="utf-8-sig")

# ---------- 7. 시각화 ----------
from matplotlib.colors import to_rgba
PALETTE = ["#FF9E9E", "#8EC9FF", "#A8E6A1", "#FFD97A", "#D4B5FF"]   # 밝은 파스텔 톤
col = np.array([to_rgba(c) for c in PALETTE[:k]]); fig = plt.figure(figsize=(19, 12)); gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])
def patch(ax, geom, **kw):
    for p in (geom.geoms if geom.geom_type == "MultiPolygon" else [geom]):
        v, c = [], []
        for ring in [p.exterior] + list(p.interiors):
            xy = np.asarray(ring.coords); v += list(xy); c += [Path.MOVETO] + [Path.LINETO] * (len(xy) - 2) + [Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(v, c), **kw))
ax = fig.add_subplot(gs[0, :2])
for u, gm in unit_geom.items():
    if u in df.index: patch(ax, gm, fc=col[df.loc[u, "군집"]], ec="#555", lw=.7, alpha=1)
    else: patch(ax, gm, fc="#ddd", ec="k", lw=.5)
for u, gm in unit_geom.items():
    r = gm.representative_point(); ax.text(r.x, r.y, u.replace("장안북부8동", "장안북부\n(8개동)"), ha="center", va="center", fontsize=7.5)
st = bs.apply(lambda r: to5179(r["경도"], r["위도"]), axis=1); ax.scatter([p[0] for p in st], [p[1] for p in st], s=1.5, c="k", alpha=.35)
b = unary_union(geoms).bounds; ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3]); ax.set_aspect("equal"); ax.axis("off")
ax.legend([plt.Rectangle((0, 0), 1, 1, fc=col[c]) for c in range(k)], [f"{cn[c]} ({(df['군집']==c).sum()}개)" for c in range(k)], loc="lower left", fontsize=9)
ax.set_title(f"① 군집 지도 (K-means k={k}, 검은 점=버스정류장 {len(bs):,}개)")

ax = fig.add_subplot(gs[0, 2]); P = PCA(2).fit(X); Z = P.transform(X)
ax.scatter(Z[:, 0], Z[:, 1], c=col[df["군집"].values], s=70, edgecolor="k")
for (u, _), (a, b_) in zip(df.iterrows(), Z): ax.annotate(u.replace("장안북부8동", "장안북부"), (a, b_), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.set_title(f"② PCA 분포 (설명분산 {P.explained_variance_ratio_.sum()*100:.0f}%)"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.grid(alpha=.25)

ax = fig.add_subplot(gs[1, :2]); im = ax.imshow(z.values, cmap="coolwarm", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats, rotation=15, ha="right", fontsize=9)
ax.set_yticks(range(k)); ax.set_yticklabels([f"{cn[c]}  (n={(df['군집']==c).sum()})" for c in z.index], fontsize=9)
for i in range(k):
    for j in range(len(feats)): ax.text(j, i, f"{prof.iloc[i, j]:.1f}", ha="center", va="center", fontsize=9)
ax.set_title("③ 군집 프로파일 (색=전체평균 대비 z, 숫자=실제 평균)"); plt.colorbar(im, ax=ax, fraction=.025)

ax = fig.add_subplot(gs[1, 2]); ct = pd.crosstab(df["구"], df["군집"]).reindex(["장안구", "권선구", "팔달구", "영통구"]).fillna(0); bot = np.zeros(4)
for c in range(k):
    ax.bar(ct.index, ct[c], bottom=bot, color=col[c], edgecolor="k", lw=.5, label=f"C{c+1}")
    for i, v in enumerate(ct[c]):
        if v: ax.text(i, bot[i] + v / 2, int(v), ha="center", va="center", color="#222", fontweight="bold")
    bot += ct[c].values
ax.set_title(f"④ 구별 군집 구성 (실루엣: k=3 {sil[3]:.2f}/4 {sil[4]:.2f}/5 {sil[5]:.2f})"); ax.legend(fontsize=8)
fig.suptitle("수원시 교통약자 이동유형 군집분석 v2 – 행정동 단위 (장애유형·정류장 접근성·고령/독거)", fontsize=16, y=.995)
plt.tight_layout(); plt.savefig(OUT + "suwon_dong_cluster_v2_result.png", dpi=125)

pd.set_option("display.width", 220)
print("실루엣", {i: round(v, 3) for i, v in sil.items()}, "→ k =", k)
print(prof.round(1).to_string())
for c in range(k): print(cn[c], "|", ", ".join(df[df["군집"] == c].index))
print(df[["구", "장애인수", "정류장수", "면적㎢", "정류장밀도(개/㎢)", "장애인/정류장(명)"]].round(1).sort_values("장애인/정류장(명)", ascending=False).head(8).to_string())
print("상관:\n", df[feats].corr().round(2).to_string())
