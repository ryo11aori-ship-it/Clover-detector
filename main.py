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

    # --- STEP 1: 画像を「丸探し」用に加工 ---
    # グレースケール化
    # 普通の gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ではなく、
    # 「緑成分」を強調するために Gチャンネルと Rチャンネルの差分を使ってみる
    b, g, r = cv2.split(img)
    
    # 緑 - 赤 を計算することで、緑の部分が白く、それ以外が暗くなる
    # これで葉っぱのコントラストを稼ぐ
    green_diff = cv2.subtract(g, r)
    
    # ノイズ除去（強め）
    blurred = cv2.GaussianBlur(green_diff, (9, 9), 2)
    
    # --- STEP 2: ハフ変換で「円」を検出 ---
    # ここが魔術。画像内の「丸い成分」を総当たりで探す。
    # param1: Cannyのエッジ検出閾値（高いと減る）
    # param2: 円の中心検出の閾値（低いと誤検出増える、高いと厳しくなる）
    # minRadius/maxRadius: 葉っぱの大きさ（ピクセル）
    
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2,          # 解像度の比率
        minDist=15,      # 円同士の最小距離（葉っぱは密集してるので小さめに）
        param1=50,       
        param2=25,       # ここを調整して検出数を制御（20〜30あたりが怪しい）
        minRadius=8,     # 小さすぎるゴミは無視
        maxRadius=45     # デカすぎる円も無視
    )

    valid_circles = []

    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(f"Circles detected: {len(circles[0, :])}")
        
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            
            # --- STEP 3: 色チェック ---
            # 丸が見つかっても、それが「緑色」じゃなかったら意味がない（白い石とか）
            # 円の中心の色を取得して確認
            if 0 <= center[1] < img.shape[0] and 0 <= center[0] < img.shape[1]:
                pixel_color = img[center[1], center[0]]
                # 緑成分が他より強いか簡易チェック (G > R and G > B)
                if pixel_color[1] > pixel_color[0] and pixel_color[1] > pixel_color[2]:
                    valid_circles.append({'x': i[0], 'y': i[1], 'r': i[2]})
                    
                    # 青い丸で検出された「葉っぱ」を描画（デバッグ）
                    cv2.circle(output_img, center, radius, (255, 0, 0), 1)
                    cv2.circle(output_img, center, 2, (0, 255, 255), 3)

    print(f"Green circles kept: {len(valid_circles)}")

    # --- STEP 4: 幾何学的クラスタリング ---
    # 「ある円の近くに、似たようなサイズの円があと3つあるか？」
    
    found_clovers = []
    
    for i, c1 in enumerate(valid_circles):
        neighbors = []
        search_dist = c1['r'] * 3.5  # 半径の3.5倍以内を探す
        
        for j, c2 in enumerate(valid_circles):
            if i == j: continue
            
            dist = np.sqrt((c1['x'] - c2['x'])**2 + (c1['y'] - c2['y'])**2)
            
            # 距離が近く、かつサイズも極端に違わないもの
            if dist < search_dist and 0.5 < (c1['r'] / c2['r']) < 2.0:
                neighbors.append(c2)
        
        # 自分(1) + 近傍(3) = 4つ のセット
        # 4〜5個の密集があれば四葉の可能性が高い
        if 3 <= len(neighbors) <= 4:
            # 重心を計算
            all_x = [c1['x']] + [n['x'] for n in neighbors]
            all_y = [c1['y']] + [n['y'] for n in neighbors]
            cx = int(sum(all_x) / len(all_x))
            cy = int(sum(all_y) / len(all_y))
            
            # 重複排除
            is_new = True
            for fc in found_clovers:
                if np.sqrt((fc[0]-cx)**2 + (fc[1]-cy)**2) < 20:
                    is_new = False
                    break
            
            if is_new:
                found_clovers.append((cx, cy))

    # --- STEP 5: 結果描画 ---
    if len(found_clovers) > 0:
        for cx, cy in found_clovers:
            print(f"CLOVER at ({cx}, {cy})")
            cv2.circle(output_img, (cx, cy), 40, (0, 0, 255), 4)
            cv2.putText(output_img, "4-LEAF!!", (cx-30, cy-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        print("No luck. Trying to highlight dense areas.")
        # 見つからなくても、とりあえず一番丸が集まってる場所を出す
        if len(valid_circles) > 0:
            best_c = valid_circles[0]
            max_n = 0
            for c1 in valid_circles:
                n_count = 0
                for c2 in valid_circles:
                    if np.sqrt((c1['x']-c2['x'])**2 + (c1['y']-c2['y'])**2) < 50:
                        n_count += 1
                if n_count > max_n:
                    max_n = n_count
                    best_c = c1
            
            cv2.circle(output_img, (best_c['x'], best_c['y']), 50, (0, 255, 255), 2)
            cv2.putText(output_img, "Here?", (best_c['x']-20, best_c['y']), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imwrite("clover_marked.png", output_img)
    print("Saved.")

if __name__ == "__main__":
    main()
