import cv2
import numpy as np
import sys
import os
import math

# エラーハンドリングとメイン処理を一箇所にまとめる
def main():
    # 入力ファイルの取得
    if len(sys.argv) < 2: print("Usage: clover_finder.exe <image_path>"); return
    input_path = sys.argv[1]
    if not os.path.exists(input_path): print(f"Error: File not found {input_path}"); return
    
    # 画像読み込み
    img = cv2.imread(input_path)
    if img is None: print("Error: Failed to load image."); return
    output_img = img.copy()
    
    # STEP 1: 前処理 (HSV変換と緑色抽出)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 緑色の範囲 (色相H: 35-85, 彩度S: 40-255, 明度V: 40-255)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # ノイズ除去 (モルフォロジー演算: オープニング -> クロージング)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # STEP 2: 輪郭抽出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 葉っぱ候補のリスト化 (重心と面積)
    leaf_candidates = []
    min_area = 50 # 小さすぎるノイズを除去
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area: continue
        
        # 円形度計算 (葉っぱらしさの判定用だが、今回は緩めに採用)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        
        # 重心計算
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        # 候補として保存 (x, y, area, contour)
        leaf_candidates.append({'x': cX, 'y': cY, 'area': area, 'cnt': cnt})

    print(f"Leaf candidates found: {len(leaf_candidates)}")

    # STEP 3 & 4: クラスタリングと四葉判定
    # シンプルに「ある葉っぱから一定距離内に、自分を含めて4枚の葉っぱがあれば四葉」とみなす
    detected_clovers = []
    search_radius_ratio = 4.0 # 葉っぱの大きさに対する探索半径の比率
    
    # 重複描画を防ぐためのマスク
    marked_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for i, leaf in enumerate(leaf_candidates):
        # この葉っぱを基準に、近くに何個あるか探す
        # 基準半径 = sqrt(area) 近似
        radius_proxy = math.sqrt(leaf['area'])
        search_dist = radius_proxy * search_radius_ratio
        
        neighbors = []
        for j, target in enumerate(leaf_candidates):
            dist = math.sqrt((leaf['x'] - target['x'])**2 + (leaf['y'] - target['y'])**2)
            if dist < search_dist: neighbors.append(target)
        
        # STEP 5: 判定ロジック (枚数が4枚、かつ面積のバラつきが極端でないこと)
        if len(neighbors) == 4:
            areas = [n['area'] for n in neighbors]
            max_a = max(areas)
            min_a = min(areas)
            
            # 面積比チェック (極端に大きい/小さい葉が混じってないか)
            if min_a / max_a > 0.2: 
                # クラスタの重心を計算
                avg_x = int(sum([n['x'] for n in neighbors]) / 4)
                avg_y = int(sum([n['y'] for n in neighbors]) / 4)
                
                # すでにマーク済みの場所に近い場合はスキップ (重複排除)
                if marked_mask[avg_y, avg_x] == 0:
                    # STEP 6: 赤丸描画
                    # 半径はクラスタの広がり具合から決定
                    max_dist_from_center = 0
                    for n in neighbors:
                        d = math.sqrt((avg_x - n['x'])**2 + (avg_y - n['y'])**2)
                        if d > max_dist_from_center: max_dist_from_center = d
                    
                    draw_radius = int(max_dist_from_center * 1.5)
                    cv2.circle(output_img, (avg_x, avg_y), draw_radius, (0, 0, 255), 5)
                    
                    # マスクを更新 (近傍を埋める)
                    cv2.circle(marked_mask, (avg_x, avg_y), draw_radius, 255, -1)
                    print(f"Found a potential 4-leaf clover at ({avg_x}, {avg_y})")

    # 結果保存
    filename, ext = os.path.splitext(input_path)
    output_path = f"{filename}_marked.png"
    cv2.imwrite(output_path, output_img)
    print(f"Done. Saved to {output_path}")

if __name__ == "__main__":
    main()
