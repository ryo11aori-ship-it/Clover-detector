import cv2
import numpy as np
import sys
import os

def main():
    # --- 入力処理 ---
    if len(sys.argv) < 2: 
        print("Usage: exe <img_path>")
        return
    input_path = sys.argv[1]
    if not os.path.exists(input_path): 
        print(f"Error: File not found {input_path}")
        return
    
    img = cv2.imread(input_path)
    if img is None: return
    output_img = img.copy()

    # --- STEP 1: 緑色の抽出（より繊細に） ---
    # ブラーを弱める（葉の境目を消さないため）
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # 緑マスク（少し暗めも拾う設定）
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # ノイズ処理：小さな穴は埋めるが、葉の間の隙間（黒い線）は埋めないように注意
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- STEP 2: 距離変換 ---
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    
    # --- STEP 3: ローカルマキシマ（局所的な山頂）を探す ---
    # ここが今回のキモ。全体基準ではなく「隣より高いか」を見る
    
    # 1. 距離画像を膨張させる（各ピクセルに周辺の最大値を代入）
    dilated = cv2.dilate(dist_transform, np.ones((3,3), np.uint8))
    
    # 2. 「元の値 == 膨張後の値」の場所が、周辺で一番高い場所（ピーク）
    local_peaks = (dist_transform == dilated)
    
    # 3. ノイズ除去：ピークの中でも、ある程度「葉っぱっぽい厚み」があるものだけ残す
    # 閾値を下げて（0.1倍）、平べったい葉っぱも拾う
    peak_threshold = 0.1 * dist_transform.max()
    peaks_mask = local_peaks & (dist_transform > peak_threshold)
    
    # ピーク座標の取得
    # np.where は (y_array, x_array) を返す
    y_peaks, x_peaks = np.where(peaks_mask)
    
    # 座標リスト化 (x, y)
    leaf_centers = list(zip(x_peaks, y_peaks))
    
    print(f"Leaf peaks found: {len(leaf_centers)}")

    # 青ドット描画（デバッグ用：今度は四つ葉の上にも出るはず！）
    for x, y in leaf_centers:
        cv2.circle(output_img, (x, y), 2, (255, 0, 0), -1)

    # --- STEP 4: 密集判定（近傍4点ロジック） ---
    # 判定を少し甘くする（四角形っぽくなくても、近くにあればOK）
    
    search_radius = 40  # 画素数固定（画像の解像度に合わせて調整が必要かも）
    clover_candidates = []

    for i in range(len(leaf_centers)):
        c1 = leaf_centers[i]
        neighbors = []
        
        for j in range(len(leaf_centers)):
            c2 = leaf_centers[j]
            d = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
            if d < search_radius:
                neighbors.append(c2)
        
        # 自分を含めて4つ以上あれば候補
        if len(neighbors) >= 4:
            # 重心を計算
            cx = int(sum([n[0] for n in neighbors]) / len(neighbors))
            cy = int(sum([n[1] for n in neighbors]) / len(neighbors))
            
            # 重複チェック
            is_new = True
            for existing in clover_candidates:
                if np.sqrt((existing[0]-cx)**2 + (existing[1]-cy)**2) < 20:
                    is_new = False
                    break
            
            if is_new:
                clover_candidates.append((cx, cy))

    # --- STEP 5: 結果描画 ---
    if len(clover_candidates) > 0:
        for cx, cy in clover_candidates:
            print(f"Lucky Clover at ({cx}, {cy})")
            # 派手にマーク
            cv2.circle(output_img, (cx, cy), 40, (0, 0, 255), 4)
            cv2.putText(output_img, "4-LEAF!", (cx-30, cy-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        print("No 4-leaf found... detecting 3-leaf clusters instead.")
        # ボウズ回避：3つ組を探して黄色で出す
        for i in range(len(leaf_centers)):
            c1 = leaf_centers[i]
            neighbors = [c2 for c2 in leaf_centers if np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) < search_radius]
            if len(neighbors) == 3:
                cx = int(sum([n[0] for n in neighbors]) / 3)
                cy = int(sum([n[1] for n in neighbors]) / 3)
                cv2.circle(output_img, (cx, cy), 30, (0, 255, 255), 2)

    # 保存
    cv2.imwrite("clover_marked.png", output_img)
    print("Saved clover_marked.png")

if __name__ == "__main__":
    main()
