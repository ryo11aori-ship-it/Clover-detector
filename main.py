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

    # --- STEP 1: ノイズを抹殺する ---
    # 強めのブラーをかけて、葉脈や模様を消し去る
    blurred = cv2.medianBlur(img, 15)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # 緑色マスク (かなり限定的に)
    lower_green = np.array([35, 60, 60])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # --- STEP 2: 侵食（Erosion）で分離 ---
    # ここが今回の肝。「葉っぱをガリガリ削って小さくする」
    # これにより、隣接する葉っぱとの結合を切断する
    kernel = np.ones((5,5), np.uint8)
    
    # iterationsの回数だけ削る。この回数が重要。
    # 密集地帯なので、思いっきり削って「芯」だけ残す作戦
    eroded_mask = cv2.erode(mask, kernel, iterations=8)
    
    # 削りすぎて消えちゃった部分を少しだけ戻す（膨張）して形を整える
    # ただし、戻しすぎるとまたくっつくので控えめに
    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=2)

    # デバッグ用にマスク画像を保存（どう認識されてるか確認用）
    cv2.imwrite("debug_mask.png", dilated_mask)

    # --- STEP 3: 芯（重心）を見つける ---
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    leaf_centers = []
    min_area = 20 # 削りすぎたゴミは無視
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                leaf_centers.append((cX, cY))
                
    print(f"Leaf cores found: {len(leaf_centers)}")
    
    # 青ドット（芯）を描画
    for x, y in leaf_centers:
        cv2.circle(output_img, (x, y), 3, (255, 0, 0), -1)

    # --- STEP 4: クールなクラスタリング ---
    # 「半径R以内に自分を含めて4つ芯があるか？」
    
    search_radius = 55 # 画像サイズに合わせて微調整した値
    found_groups = []

    for i, p1 in enumerate(leaf_centers):
        neighbors = []
        for j, p2 in enumerate(leaf_centers):
            dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            if dist < search_radius:
                neighbors.append(p2)
        
        if len(neighbors) == 4:
            # 4つの重心
            cx = int(sum([n[0] for n in neighbors]) / 4)
            cy = int(sum([n[1] for n in neighbors]) / 4)
            
            # 形状チェック（正方形に近いか？）
            # 重心からの距離の分散を見ることで、細長い誤検出を防ぐ
            dists_from_center = [np.sqrt((n[0]-cx)**2 + (n[1]-cy)**2) for n in neighbors]
            dist_std = np.std(dists_from_center)
            
            # バラつきが少ない（＝綺麗な放射状）場合のみ採用
            if dist_std < 15:
                # 重複排除
                is_new = True
                for g in found_groups:
                    if np.sqrt((g[0]-cx)**2 + (g[1]-cy)**2) < 20:
                        is_new = False
                        break
                if is_new:
                    found_groups.append((cx, cy))

    # --- STEP 5: 描画 ---
    if len(found_groups) > 0:
        for cx, cy in found_groups:
            print(f"FOUND at ({cx}, {cy})")
            cv2.circle(output_img, (cx, cy), 50, (0, 0, 255), 4)
            cv2.putText(output_img, "FOUND!", (cx-40, cy-40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        # 万が一見つからなかったら「一番それっぽい場所」を推定
        print("Strict mode failed. Guessing...")
        max_neighbors = 0
        best_spot = None
        for i, p1 in enumerate(leaf_centers):
             neighbors = [p2 for p2 in leaf_centers if np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) < search_radius]
             if len(neighbors) > max_neighbors:
                 max_neighbors = len(neighbors)
                 cx = int(sum([n[0] for n in neighbors]) / len(neighbors))
                 cy = int(sum([n[1] for n in neighbors]) / len(neighbors))
                 best_spot = (cx, cy)
        
        if best_spot:
             cv2.circle(output_img, best_spot, 50, (0, 255, 255), 3)
             cv2.putText(output_img, f"Maybe? ({max_neighbors})", (best_spot[0]-40, best_spot[1]-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imwrite("clover_marked.png", output_img)
    print("Saved.")

if __name__ == "__main__":
    main()
