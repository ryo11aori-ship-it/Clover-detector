#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fourleaf_finder v2.0
====================
古典的コンピュータビジョンの限界まで詰めた「四つ葉のクローバー」検出器。

パイプライン:
 1. 前処理            : グレーワールド色補正 + CLAHE(局所コントラスト補正)
 2. 植生セグメンテーション: HSV緑域 / ExG(超過緑指数+Otsu) / Lab a* の3手法の多数決
 3. 候補生成(2系統)
    A) 孤立株解析     : 連結成分ごとに距離変換最大点を中心として放射シグネチャ r(θ) を取り、
                        FFT 4次高調波の優勢度・ローブ(葉)数・角度等間隔性・90°回転対称性で判定
    B) 群生地解析     : 距離変換の局所極大 = 小葉(リーフレット)候補を検出し、
                        「対角ペアの中点が一致する 4 小葉の十字配置」を幾何学的に照合。
                        密生したクローバー畑(連結成分が巨大な塊になる状況)でも動作する。
 4. 検証スコアリング   : 角度等間隔性 / 小葉半径の一様性 / 中心距離(リング)の一様性 /
                        小葉間の谷(葉境界)の存在 / 4回回転対称性(90° IoU vs 120° IoU)
                        を重み付き合成して信頼度 0-1 を算出
 5. NMS + 注釈       : 重複検出を除去し、元解像度の画像へ信頼度付きでマーキング

限界について(重要):
  三つ葉との弁別を古典CVだけで「保証」することは原理的に不可能です(遮蔽・重なり・
  葉の欠け・5小葉の存在など)。本実装は「4ローブの回転対称構造」を独立した複数の
  証拠で検証して誤検出を最小化する設計であり、信頼度スコアで判断材料を提示します。
  さらに上を狙う場合はアノテーション済みデータで学習した検出器が必要です。
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# I/O ユーティリティ(Windows の日本語パス対応)
# ---------------------------------------------------------------------------
def imread_unicode(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True


def parse_args():
    p = argparse.ArgumentParser(
        description="High-accuracy four-leaf clover detector (classical CV, v2)")
    p.add_argument("image", help="input image path")
    p.add_argument("-o", "--out", default="clover_marked.png",
                   help="output annotated image path")
    p.add_argument("--min-conf", type=float, default=0.60,
                   help="confidence threshold 0-1 (default 0.60)")
    p.add_argument("--min-leaf-r", type=int, default=0,
                   help="min leaflet radius in px after resize (0 = auto)")
    p.add_argument("--max-dim", type=int, default=1800,
                   help="processing resolution: longest side (default 1800)")
    p.add_argument("--debug", action="store_true",
                   help="save intermediate images (debug_*.png)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 前処理
# ---------------------------------------------------------------------------
def gray_world_wb(img):
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)
    gray = means.mean()
    for c in range(3):
        f[:, :, c] *= gray / max(means[c], 1e-6)
    return np.clip(f, 0, 255).astype(np.uint8)


def clahe_enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def vegetation_mask(img):
    """HSV / ExG / Lab a* の 3 手法の多数決で頑健な植生マスクを作る。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m_hsv = cv2.inRange(hsv, (30, 35, 25), (95, 255, 255))

    f = img.astype(np.float32)
    b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    s = b + g + r + 1e-6
    exg = 2.0 * g / s - r / s - b / s            # 正規化超過緑指数
    exg8 = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m_exg = cv2.threshold(exg8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    m_lab = cv2.inRange(lab[:, :, 1], 0, 125)    # a* < 125 → 緑寄り

    vote = (m_hsv // 255).astype(np.uint8) \
         + (m_exg // 255).astype(np.uint8) \
         + (m_lab // 255).astype(np.uint8)
    mask = np.where(vote >= 2, 255, 0).astype(np.uint8)

    k = max(3, (min(img.shape[:2]) // 400) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# ---------------------------------------------------------------------------
# 信号処理ユーティリティ(scipy 非依存)
# ---------------------------------------------------------------------------
def smooth_circular(sig, k=9):
    k = max(3, int(k) | 1)
    kernel = cv2.getGaussianKernel(k, -1).flatten()
    ext = np.concatenate([sig[-k:], sig, sig[:k]])
    sm = np.convolve(ext, kernel, mode="same")
    return sm[k:-k]


def circular_peaks(sig, prom_ratio=0.12, min_sep_deg=35):
    """循環信号のピークを卓立度(prominence)付きで検出する。"""
    n = len(sig)
    rng = float(sig.max() - sig.min())
    if rng < 1e-6:
        return []
    cand = [i for i in range(n)
            if sig[i] > sig[(i - 1) % n] and sig[i] >= sig[(i + 1) % n]]
    peaks = []
    for i in cand:
        j, steps = i, 0
        while steps < n:
            nj = (j - 1) % n
            if sig[nj] > sig[j]:
                break
            j, steps = nj, steps + 1
        left_v = sig[j]
        j, steps = i, 0
        while steps < n:
            nj = (j + 1) % n
            if sig[nj] > sig[j]:
                break
            j, steps = nj, steps + 1
        right_v = sig[j]
        prom = sig[i] - max(left_v, right_v)
        # 相対卓立度に加え、信号平均に対する絶対卓立度も要求
        # (ほぼ平坦な信号のノイズ極大をピークと誤認しないため)
        if prom >= prom_ratio * rng and prom >= 0.08 * float(np.mean(sig)):
            peaks.append((i, float(sig[i])))
    # 近接ピークのマージ(値の大きい順に貪欲選択)
    peaks.sort(key=lambda t: -t[1])
    min_sep = min_sep_deg * n / 360.0
    kept = []
    for i, v in peaks:
        if all(min(abs(i - j), n - abs(i - j)) >= min_sep for j, _ in kept):
            kept.append((i, v))
    return kept


def angle_regularity(angles_deg, k):
    """k 個の角度が 360/k ° 等間隔にどれだけ近いか(0-1)。"""
    a = sorted(a % 360 for a in angles_deg)
    gaps = [(a[(i + 1) % k] - a[i]) % 360 for i in range(k)]
    ideal = 360.0 / k
    dev = float(np.std(gaps))
    return float(np.clip(1.0 - dev / (0.5 * ideal), 0.0, 1.0))


def fourier_four_score(sig):
    """放射シグネチャの 4 次高調波優勢度(四つ葉らしさ)0-1。"""
    s = sig - sig.mean()
    F = np.abs(np.fft.rfft(s))
    if len(F) < 9:
        return 0.0
    band = float(F[2:9].sum()) + 1e-9
    dom = float(F[4]) / band
    rival = max(float(F[3]), float(F[5]))
    margin = (float(F[4]) - rival) / (float(F[4]) + rival + 1e-9)   # -1..1
    return float(np.clip(0.5 * min(dom * 3.0, 1.0) + 0.5 * (0.5 + 0.5 * margin),
                         0.0, 1.0))


def radial_signature(mask, cx, cy, rmax, n=180):
    """中心 (cx,cy) から放射方向にレイキャストして輪郭半径 r(θ) を得る。"""
    H, W = mask.shape
    rmax = max(int(rmax), 4)
    sig = np.zeros(n, np.float32)
    gap_tol = max(2, int(0.05 * rmax))
    for i in range(n):
        th = 2.0 * math.pi * i / n
        c, s = math.cos(th), math.sin(th)
        r_hit, gap = 0, 0
        for r in range(1, rmax):
            x = int(round(cx + r * c))
            y = int(round(cy + r * s))
            if 0 <= x < W and 0 <= y < H and mask[y, x]:
                r_hit, gap = r, 0
            else:
                gap += 1
                if gap > gap_tol:
                    break
        sig[i] = r_hit
    return sig


# ---------------------------------------------------------------------------
# 回転対称性
# ---------------------------------------------------------------------------
def disk_crop(mask, cx, cy, R):
    """(cx,cy) を中心とした半径 R の円板内マスクを正方キャンバスに切り出す。"""
    R = max(int(R), 4)
    size = 2 * R + 1
    canvas = np.zeros((size, size), np.uint8)
    H, W = mask.shape
    x0, y0 = int(round(cx)) - R, int(round(cy)) - R
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x0 + size), min(H, y0 + size)
    if sx1 <= sx0 or sy1 <= sy0:
        return canvas
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = mask[sy0:sy1, sx0:sx1]
    yy, xx = np.ogrid[:size, :size]
    canvas[(xx - R) ** 2 + (yy - R) ** 2 > R * R] = 0
    return canvas


def rot_iou(local, k):
    h, w = local.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 360.0 / k, 1.0)
    rot = cv2.warpAffine(local, M, (w, h), flags=cv2.INTER_NEAREST)
    a = local > 0
    b = rot > 0
    union = np.logical_or(a, b).sum()
    if union < 20:
        return 0.0
    return float(np.logical_and(a, b).sum()) / float(union)


def symmetry_score(local):
    """90°回転対称(四つ葉)が 120°回転対称(三つ葉)より強いほど高スコア。"""
    s4 = rot_iou(local, 4)
    s3 = rot_iou(local, 3)
    return float(np.clip((s4 - s3) * 4.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 候補生成 A: 孤立株(連結成分)解析
# ---------------------------------------------------------------------------
def component_detections(mask, r_min):
    dets = []
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    H, W = mask.shape
    min_area = 8 * r_min * r_min
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area or area > 0.25 * H * W:
            continue
        comp = np.zeros((h + 2, w + 2), np.uint8)
        comp[1:-1, 1:-1] = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        d = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        maxd = float(d.max())
        if maxd < r_min:
            continue
        # 面積が単一株として説明できないほど大きい塊 → 群生地パスに任せる
        if area > 7.0 * math.pi * maxd * maxd:
            continue
        cy0, cx0 = np.unravel_index(int(np.argmax(d)), d.shape)
        req = int(round(2.6 * math.sqrt(area / math.pi)))
        sig = radial_signature(comp, cx0, cy0, req)
        sm = smooth_circular(sig, 9)
        if sm.max() <= max(2.0, 0.6 * r_min):
            continue
        # 変調深度: ローブ構造なら葉間の谷が深い。平坦(円盤・塊)は棄却。
        mod = float((sm.max() - sm.min()) / max(sm.mean(), 1e-9))
        if mod < 0.30:
            continue
        pk = circular_peaks(sm)
        kcnt = len(pk)
        if kcnt == 3:
            continue                       # 三つ葉として棄却
        four = fourier_four_score(sm) * float(min(1.0, mod / 0.5))
        peak_term = 1.0 if kcnt == 4 else (0.25 if kcnt == 5 else 0.0)
        if kcnt == 4:
            n = len(sm)
            angs = [p[0] * 360.0 / n for p in pk]
            ang = angle_regularity(angs, 4)
            hv = [p[1] for p in pk]
            runif = float(np.clip(
                1.0 - (np.std(hv) / (np.mean(hv) + 1e-9)) / 0.35, 0.0, 1.0))
        else:
            ang, runif = 0.0, 0.0
        sym = symmetry_score(disk_crop(comp, cx0, cy0, int(req * 0.95)))
        conf = 0.30 * four + 0.25 * peak_term + 0.15 * ang \
             + 0.10 * runif + 0.20 * sym
        if conf <= 0.05:
            continue
        disp_r = float(np.mean([p[1] for p in pk])) if pk else 2.0 * maxd
        dets.append(dict(x=float(x + cx0 - 1), y=float(y + cy0 - 1),
                         r=max(disp_r, 3.0 * r_min),
                         conf=float(np.clip(conf, 0, 1)), method="isolated"))
    return dets


# ---------------------------------------------------------------------------
# 候補生成 B: 群生地の小葉クラスタ(ロゼット)解析
# ---------------------------------------------------------------------------
def leaflet_maxima(dist, r_min, cap=1500):
    """距離変換の局所極大 = 小葉中心の候補 (x, y, 小葉半径)。"""
    k = max(3, (2 * r_min) | 1)
    dil = cv2.dilate(dist, np.ones((k, k), np.uint8))
    peaks = (dist >= r_min) & (dist >= dil - 1e-4)
    ys, xs = np.nonzero(peaks)
    pts = sorted(((float(x), float(y), float(dist[y, x]))
                  for x, y in zip(xs, ys)), key=lambda t: -t[2])
    if len(pts) > 12000:
        pts = pts[:12000]
    kept = []
    for x, y, r in pts:
        ok = True
        for X, Y, R in kept:
            # 既存極大の円板コア内部にある候補のみ抑制する
            # (隣接する別の小葉の極大まで消さないよう max 半径基準×0.95)
            sep = 0.95 * max(r, R)
            if (x - X) ** 2 + (y - Y) ** 2 < sep * sep:
                ok = False
                break
        if ok:
            kept.append((x, y, r))
            if len(kept) >= cap:
                break
    return kept


def verify_rosette(mask, dist, leaflets, cx, cy):
    """候補中心 (cx,cy) の周囲が「四つ葉の十字配置」かを幾何検証する。"""
    H, W = mask.shape
    icx, icy = int(round(cx)), int(round(cy))
    if not (0 <= icx < W and 0 <= icy < H):
        return None
    # 中心近傍(±3px)に植生があること
    if mask[max(0, icy - 3):icy + 4, max(0, icx - 3):icx + 4].max() == 0:
        return None
    # 中心接合部そのものの距離変換極大は「小葉」ではないので除外する
    # (中心からの距離が自身の半径以下 = 中心を覆うブロブ)
    dd = sorted(((math.hypot(l[0] - cx, l[1] - cy), l) for l in leaflets
                 if math.hypot(l[0] - cx, l[1] - cy) > 0.9 * l[2]),
                key=lambda t: t[0])
    if len(dd) < 4:
        return None
    # 近傍候補から半径外れ値(ノイズ極大)を除去してから最近傍4点を選ぶ
    med = float(np.median([t[1][2] for t in dd[:8]]))
    cand = [t for t in dd[:12] if 0.5 * med <= t[1][2] <= 2.0 * med]
    if len(cand) < 4:
        return None
    m4 = cand[:4]
    # 4小葉の重心で中心を再推定(候補中心のズレに頑健にする)
    cx = float(np.mean([t[1][0] for t in m4]))
    cy = float(np.mean([t[1][1] for t in m4]))
    # 再推定中心の近傍(±3px)にも植生があること
    icx, icy = int(round(cx)), int(round(cy))
    if not (0 <= icx < W and 0 <= icy < H):
        return None
    if mask[max(0, icy - 3):icy + 4, max(0, icx - 3):icx + 4].max() == 0:
        return None
    m4 = [(math.hypot(t[1][0] - cx, t[1][1] - cy), t[1]) for t in m4]
    dists = [t[0] for t in m4]
    rho = float(np.mean(dists))                     # リング半径(中心→小葉)
    if rho < 3.0:
        return None
    radii = [t[1][2] for t in m4]
    rleaf = float(np.mean(radii))
    if max(radii) / max(min(radii), 1e-6) > 2.2:    # 小葉サイズが不揃いすぎ
        return None
    if rho > 3.2 * rleaf or rho < 0.8 * rleaf:      # クローバーの比率から逸脱
        return None
    angs = sorted(math.degrees(math.atan2(t[1][1] - cy, t[1][0] - cx)) % 360
                  for t in m4)
    gaps = [(angs[(i + 1) % 4] - angs[i]) % 360 for i in range(4)]
    if min(gaps) < 45 or max(gaps) > 135:           # 十字配置ではない
        return None

    ang = angle_regularity(angs, 4)
    ring = float(np.clip(1.0 - (np.std(dists) / (rho + 1e-9)) / 0.30, 0, 1))
    runif = float(np.clip(1.0 - (np.std(radii) / (rleaf + 1e-9)) / 0.35, 0, 1))

    # --- 密生地帯での「株間の偽十字」を棄却する決定的ゲート ---
    # 本物の四つ葉は (a) 中心に葉柄接合部があり距離変換が太い,
    #                (b) 4小葉が中心から正確に等距離(リング半径が揃う)。
    # 隣接する別株の小葉が偶然十字を成す偽candは中心が背景の隙間(dist≈0)で
    # かつ距離がバラつく。合成・実写どちらでもこの2軸は明瞭に分離する。
    center_thickness = float(dist[icy, icx]) / (rleaf + 1e-9)
    ring_disp = float(np.std(dists) / (rho + 1e-9))
    if center_thickness < 0.15 or ring_disp > 0.16:
        return None

    # 隣接小葉間に「谷」(葉境界で距離変換が細る)があるか
    vals = []
    for i in range(4):
        mid = math.radians(angs[i] + gaps[i] / 2.0)
        px = int(round(cx + rho * math.cos(mid)))
        py = int(round(cy + rho * math.sin(mid)))
        v = float(dist[py, px]) if (0 <= px < W and 0 <= py < H) else rleaf
        vals.append(v)
    valley = float(np.clip(1.0 - np.mean(vals) / (0.9 * rleaf + 1e-9), 0, 1))

    # 5番目以降の「同等サイズの」小葉が近すぎる = 密集地帯の誤合成の疑い → 減点
    # (微小なノイズ極大は数えない。再推定中心からの距離で判定)
    m4_ids = {id(t[1]) for t in m4}
    rest = sorted(math.hypot(l[0] - cx, l[1] - cy) for l in leaflets
                  if id(l) not in m4_ids and l[2] >= 0.45 * rleaf
                  and math.hypot(l[0] - cx, l[1] - cy) > 0.9 * l[2])
    fifth = 0.0
    if len(rest) >= 1 and rest[0] < 1.30 * rho:
        fifth += 0.25
    if len(rest) >= 2 and rest[1] < 1.30 * rho:
        fifth += 0.25

    sym = symmetry_score(disk_crop(mask, cx, cy, int(round(rho + 1.3 * rleaf))))

    conf = 0.22 * ang + 0.15 * ring + 0.13 * runif \
         + 0.25 * valley + 0.25 * sym - fifth
    if conf <= 0.05:
        return None
    return dict(x=float(cx), y=float(cy), r=float(rho + rleaf),
                conf=float(np.clip(conf, 0, 1)), method="rosette")


def rosette_detections(mask, dist, leaflets):
    dets = []
    n = len(leaflets)
    if n < 4:
        return dets
    P = np.array([[l[0], l[1]] for l in leaflets], np.float32)
    R = np.array([l[2] for l in leaflets], np.float32)
    diff = P[:, None, :] - P[None, :, :]
    D = np.sqrt((diff * diff).sum(-1))
    RM = (R[:, None] + R[None, :]) * 0.5
    ratio = R[:, None] / np.maximum(R[None, :], 1e-6)
    ok = (D >= 2.0 * RM) & (D <= 5.2 * RM) & (ratio >= 0.5) & (ratio <= 2.0)
    ok = np.triu(ok, 1)
    ii, jj = np.nonzero(ok)
    if len(ii) == 0:
        return dets
    if len(ii) > 60000:                              # 極端な密集時の安全弁
        sel = np.random.default_rng(0).choice(len(ii), 60000, replace=False)
        ii, jj = ii[sel], jj[sel]
    mids = (P[ii] + P[jj]) * 0.5

    # 「対角ペアの中点」をグリッドで集約 → 支持数の多いセルほど有力な中心
    cell = float(max(4.0, np.median(R)))
    buckets = {}
    for mx, my in mids:
        key = (int(mx // cell), int(my // cell))
        b = buckets.setdefault(key, [0, 0.0, 0.0])
        b[0] += 1
        b[1] += float(mx)
        b[2] += float(my)
    cands = sorted(((b[0], b[1] / b[0], b[2] / b[0])
                    for b in buckets.values()), key=lambda t: -t[0])
    for sup, cx, cy in cands[:800]:
        det = verify_rosette(mask, dist, leaflets, cx, cy)
        if det:
            dets.append(det)
    return dets


# ---------------------------------------------------------------------------
# NMS / 注釈
# ---------------------------------------------------------------------------
def nms(dets):
    dets = sorted(dets, key=lambda d: -d["conf"])
    kept = []
    for d in dets:
        if all(math.hypot(d["x"] - k["x"], d["y"] - k["y"])
               > 0.8 * max(d["r"], k["r"]) for k in kept):
            kept.append(d)
    return kept


def annotate(orig, dets, scale):
    out = orig.copy()
    th = max(2, orig.shape[1] // 600)
    fs = max(0.6, orig.shape[1] / 1600.0)
    for idx, d in enumerate(dets, 1):
        x = int(round(d["x"] / scale))
        y = int(round(d["y"] / scale))
        r = int(round(max(d["r"] / scale, 18) * 1.25))
        cv2.circle(out, (x, y), r, (0, 0, 255), th)
        label = "4-LEAF {:.0f}%".format(d["conf"] * 100)
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        tx, ty = max(0, x - tw // 2), max(tht + 4, y - r - 8)
        cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (0, 0, 255), th, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    if not os.path.exists(args.image):
        print("ERROR: file not found: {}".format(args.image))
        return 1
    orig = imread_unicode(args.image)
    if orig is None:
        print("ERROR: cannot read image: {}".format(args.image))
        return 1

    H0, W0 = orig.shape[:2]
    scale = min(1.0, float(args.max_dim) / max(H0, W0))
    img = cv2.resize(orig, (int(W0 * scale), int(H0 * scale)),
                     interpolation=cv2.INTER_AREA) if scale < 1.0 else orig.copy()
    H, W = img.shape[:2]

    r_min = args.min_leaf_r if args.min_leaf_r > 0 else max(4, min(H, W) // 220)
    print("[info] processing at {}x{} (scale={:.3f}), min leaflet r={}px"
          .format(W, H, scale, r_min))

    pre = clahe_enhance(gray_world_wb(img))
    mask = vegetation_mask(pre)
    cover = float((mask > 0).mean())
    print("[info] vegetation coverage: {:.1f}%".format(cover * 100))
    if cover < 0.005:
        print("[warn] almost no vegetation detected - check the input image")

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    leaflets = leaflet_maxima(dist, r_min)
    print("[info] leaflet candidates: {}".format(len(leaflets)))

    dets = component_detections(mask, r_min) + rosette_detections(mask, dist, leaflets)
    all_dets = nms(dets)
    final = [d for d in all_dets if d["conf"] >= args.min_conf]

    out = annotate(orig, final, scale)
    imwrite_unicode(args.out, out)

    print("==== RESULT ====")
    if final:
        print("{} four-leaf candidate(s) found (min-conf {:.2f})"
              .format(len(final), args.min_conf))
        for i, d in enumerate(final, 1):
            print("  #{}  (x={}, y={})  conf={:.0f}%  [{}]"
                  .format(i, int(d["x"] / scale), int(d["y"] / scale),
                          d["conf"] * 100, d["method"]))
    else:
        print("no four-leaf clover found above min-conf {:.2f}".format(args.min_conf))
        near = [d for d in all_dets if d["conf"] >= 0.35][:5]
        if near:
            print("  (near-miss candidates - try lowering --min-conf)")
            for d in near:
                print("    (x={}, y={})  conf={:.0f}%  [{}]"
                      .format(int(d["x"] / scale), int(d["y"] / scale),
                              d["conf"] * 100, d["method"]))
    print("Saved: {}".format(args.out))

    if args.debug:
        imwrite_unicode("debug_mask.png", mask)
        dn = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        imwrite_unicode("debug_dist.png", cv2.applyColorMap(dn, cv2.COLORMAP_JET))
        dbg = img.copy()
        for x, y, r in leaflets:
            cv2.circle(dbg, (int(x), int(y)), max(2, int(r)), (255, 0, 255), 1)
        for d in all_dets:
            col = (0, 0, 255) if d["conf"] >= args.min_conf else (0, 255, 255)
            cv2.circle(dbg, (int(d["x"]), int(d["y"])), int(max(d["r"], 10)), col, 2)
            cv2.putText(dbg, "{:.0f}".format(d["conf"] * 100),
                        (int(d["x"]) - 10, int(d["y"])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        imwrite_unicode("debug_candidates.png", dbg)
        print("[debug] saved debug_mask.png / debug_dist.png / debug_candidates.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
