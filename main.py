import cv2
import numpy as np
import sys
import os
import itertools

def main():
    # --- 入力処理 ---
    if len(sys.argv) < 2: print("Usage: exe <img_path>"); return
    input_path = sys.argv[1]
    if not os.path.exists(input_path): print("File not found"); return
    
    img = cv2.imread(input_path)
    if img is None: return
    output_img = img.copy()

    # --- STEP 1: 緑色の抽出 ---
    # 少しボカしてノイズ低減
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # 緑色マスク作成 (範囲は広めに)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # ノイズ除去（小さなゴミを消す）
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # --- STEP 2: 距離変換で「葉の中心」を探す ---
    # 緑色の領域内で、黒い境界からどれだけ遠いかを数値化
    # これにより、くっついた葉っぱの中心が「山頂」のように浮き出る
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # 正規化（デバッグ用・可視化のため0-255にするが計算にはdist_transformを使う）
    dist_norm = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)

    # 「確実な葉っぱの中心」だけを取り出す閾値処理
    # 画像全体の最大ピークの x% 以上の高さがある部分だけを残す
    # この調整が肝。0.3くらいに下げて、隠れた葉っぱも拾う
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # --- STEP 3: 中心点の座標を取得 ---
    # 連結成分解析で、各「山頂」の座標を得る
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_fg)
    
    print(f"Leaflets found: {num_labels - 1}")
    
    # 重心リストを作成 (背景であるインデックス0は除外)
    leaf_centers = []
    for i in range(1, num_labels):
        # 面積フィルタ：あまりに小さい点はノイズとして無視
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 10: 
            leaf_centers.append(centroids[i])

    # デバッグ：検出した全ての「葉の中心」に青い点を打つ（確認用）
    for pt in leaf_centers:
        cv2.circle(output_img, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)

    # --- STEP 4: クラスタリング（4つ密集している場所を探す） ---
    # 単純な総当たり：ある点から半径R以内に、自分を含めて「ちょうど4個」点があるか？
    
    # 探索半径の決定：
    # 葉っぱのサイズに依存するが、ここでは「画像の幅の5%」程度と仮定して動的に決める
    search_radius = img.shape[1] * 0.05 
    found_clovers = []

    for i, p1 in enumerate(leaf_centers):
        neighbors = []
        for j, p2 in enumerate(leaf_centers):
            # 距離計算
            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            if dist < search_radius:
                neighbors.append(p2)
        
        # 近傍点が4つ（自分含む）なら四葉候補！
        if len(neighbors) == 4:
            # 4点の重心を計算
            cx = int(sum([p[0] for p in neighbors]) / 4)
            cy = int(sum([p[1] for p in neighbors]) / 4)
            
            # 重複検出を防ぐ（すでに近い場所に候補があればスキップ）
            is_new = True
            for exist in found_clovers:
                if np.sqrt((exist[0]-cx)**2 + (exist[1]-cy)**2) < 20:
                    is_new = False
                    break
            
            if is_new:
                found_clovers.append((cx, cy))
                print(f"Clover found at ({cx}, {cy})")
                
                # 赤丸描画
                cv2.circle(output_img, (cx, cy), int(search_radius), (0, 0, 255), 4)
                cv2.putText(output_img, "LUCKY!", (cx-20, cy-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 1つも見つからなかった場合のやけくそ処理（3つ組をマーク）
    if len(found_clovers) == 0:
        print("No 4-leaf found. Marking 3-leaf clusters as backup.")
        for i, p1 in enumerate(leaf_centers):
            neighbors = []
            for j, p2 in enumerate(leaf_centers):
                dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                if dist < search_radius:
                    neighbors.append(p2)
            if len(neighbors) == 3:
                 cx = int(sum([p[0] for p in neighbors]) / 3)
                 cy = int(sum([p[1] for p in neighbors]) / 3)
                 # 黄色でマーク
                 cv2.circle(output_img, (cx, cy), int(search_radius), (0, 255, 255), 2)

    # --- 保存 ---
    cv2.imwrite("clover_marked.png", output_img)
    print("Saved clover_marked.png")

if __name__ == "__main__":
    main()
