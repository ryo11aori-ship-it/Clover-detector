import cv2
import numpy as np
import sys
import os

def main():
    if len(sys.argv) < 2: print("Usage: exe <img_path>"); return
    input_path = sys.argv[1]
    if not os.path.exists(input_path): print("File not found"); return
    
    img = cv2.imread(input_path)
    if img is None: return
    output_img = img.copy()

    # --- STEP 1: 画像を整える ---
    # ブラーでディテールを潰す
    blurred = cv2.medianBlur(img, 11)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # 緑色抽出
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # --- STEP 2: 適度な分離処理 ---
    # 前回の反省：8回はやりすぎた。今回は2回で止める。
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 距離変換で「中心」を探す
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # ローカルマキシマ（ピーク検出）
    dilated = cv2.dilate(dist_transform, np.ones((5,5), np.uint8))
    local_peaks = (dist_transform == dilated)
    
    # ピークの中でも、ある程度の「緑の厚み」がある場所だけ採用
    # ここの閾値調整が命。最大輝度の20%以上とする。
    peak_threshold = 0.2 * dist_transform.max()
    peaks_mask = local_peaks & (dist_transform > peak_threshold)
    
    # 座標取得
    y_peaks, x_peaks = np.where(peaks_mask)
    all_centers = list(zip(x_peaks, y_peaks))
    
    # --- STEP 3: サイズフィルタリング（超重要） ---
    # 「葉っぱサイズ」じゃないものを除外する
    valid_centers = []
    
    # 葉っぱの目安サイズ（半径ピクセル）。画像サイズから推定するか、固定値にする。
    # 今回の画像なら半径10〜40pxくらいが葉っぱ1枚のはず。
    min_leaf_radius = 5
    max_leaf_radius = 60
    
    for cx, cy in all_centers:
        # そのピーク地点での距離変換の値 ≒ 半径
        r = dist_transform[cy, cx]
        if min_leaf_radius < r < max_leaf_radius:
            valid_centers.append((cx, cy))
            # 青い点で「葉っぱ候補」を描画（デバッグ用）
            cv2.circle(output_img, (cx, cy), 2, (255, 0, 0), -1)

    print(f"Valid leaves found: {len(valid_centers)}")

    # --- STEP 4: クラスタリングと描画 ---
    # 「半径R以内に、自分を含めて4つあるか？」
    search_radius = 50  # 探索半径
    found_clovers = []

    for i, p1 in enumerate(valid_centers):
        neighbors = []
        for j, p2 in enumerate(valid_centers):
            d = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            if d < search_radius:
                neighbors.append(p2)
        
        # デバッグ：つながっているとみなした葉っぱ同士を緑の線で結ぶ
        # これで「どうグルーピングされたか」が見える
        if len(neighbors) >= 2:
            for n in neighbors:
                cv2.line(output_img, p1, n, (0, 255, 0), 1)

        # 判定：4つ（四葉）または5つ（密集した四葉）
        if 4 <= len(neighbors) <= 5:
            # 重心計算
            cx = int(sum([n[0] for n in neighbors]) / len(neighbors))
            cy = int(sum([n[1] for n in neighbors]) / len(neighbors))
            
            # 分散チェック（一直線に並んでるようなノイズを除去）
            dx = [n[0] for n in neighbors]
            dy = [n[1] for n in neighbors]
            if np.std(dx) > 5 and np.std(dy) > 5: # ある程度広がっていること
                
                # 重複排除
                is_new = True
                for f in found_clovers:
                    if np.sqrt((f[0]-cx)**2 + (f[1]-cy)**2) < 20:
                        is_new = False
                        break
                
                if is_new:
                    found_clovers.append((cx, cy))

    # --- STEP 5: 結果出力 ---
    if len(found_clovers) > 0:
        for cx, cy in found_clovers:
            print(f"CLOVER at ({cx}, {cy})")
            cv2.circle(output_img, (cx, cy), 40, (0, 0, 255), 4)
            cv2.putText(output_img, "4-LEAF", (cx-30, cy-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        # 見つからない場合は、一番密集度が高い場所をヒントとして出す
        print("No perfect 4-leaf found.")
        max_n = 0
        best_p = None
        for p1 in valid_centers:
             neighbors = [p2 for p2 in valid_centers if np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) < search_radius]
             if len(neighbors) > max_n:
                 max_n = len(neighbors)
                 cx = int(sum([n[0] for n in neighbors]) / len(neighbors))
                 cy = int(sum([n[1] for n in neighbors]) / len(neighbors))
                 best_p = (cx, cy)
        
        if best_p:
            cv2.circle(output_img, best_p, 40, (0, 255, 255), 2)
            cv2.putText(output_img, f"Dense Area ({max_n})", (best_p[0]-40, best_p[1]-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imwrite("clover_marked.png", output_img)
    print("Saved.")

if __name__ == "__main__":
    main()
