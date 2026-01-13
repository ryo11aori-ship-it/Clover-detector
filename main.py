import cv2
import numpy as np
import sys
import os

def main():
    # 入力チェック
    if len(sys.argv) < 2: 
        print("Usage: clover_finder.exe <image_path>")
        return
    input_path = sys.argv[1]
    if not os.path.exists(input_path): 
        print(f"Error: File not found {input_path}")
        return
    
    img = cv2.imread(input_path)
    if img is None: 
        print("Error: Failed to load image.")
        return
    
    output_img = img.copy()
    
    # --- STEP 1: 画像を見やすくする（前処理） ---
    # ガウシアンブラーでノイズを散らす
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # HSV変換
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # 緑色の範囲を少し広げる（影の部分も拾えるように）
    # Hue: 30-90, Saturation: 30-255, Value: 30-255
    lower_green = np.array([30, 30, 30])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 穴埋め処理（葉脈などで切れないように）
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # --- STEP 2: 輪郭抽出 ---
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    
    print(f"Total contours found: {len(contours)}")
    
    # --- STEP 3: 形状解析（凹凸を数える） ---
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # 小さすぎるゴミと、画面全体を覆うような巨大な塊は無視
        if area < 500 or area > (img.shape[0] * img.shape[1] / 2):
            continue
            
        # 輪郭を少し滑らかにする（ギザギザ対策）
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # 凸包（輪郭の外側をゴムバンドで囲んだような形）を計算
        hull = cv2.convexHull(approx, returnPoints=False)
        
        # 凸包が計算できない形状はスキップ
        if hull is None or len(hull) <= 3:
            continue
            
        try:
            # Convexity Defects（凹みの検出）
            # 輪郭と凸包の間の「隙間」を探す
            defects = cv2.convexityDefects(approx, hull)
        except:
            continue
            
        if defects is None:
            continue
            
        indentation_count = 0
        
        # 各「凹み」をチェック
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            # d は凹みの深さ（距離）
            # ある程度深い凹みだけをカウント（葉っぱの隙間とみなす）
            if d > 1000: # この閾値は画像の解像度によるが、一旦固定値で
                indentation_count += 1
        
        # 判定ロジック：凹みが4つ以上あれば「四つ葉候補」
        # クローバーの葉は丸いので、凹み（谷）の数が葉の数と相関する
        if indentation_count >= 4:
            # 重心計算
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # 外接円を取得（描画用サイズ）
                (x,y), radius = cv2.minEnclosingCircle(cnt)
                
                candidates.append({
                    'x': cX, 'y': cY, 'r': int(radius), 
                    'score': indentation_count # 凹みが多いほど複雑＝四葉っぽい？
                })

    # --- STEP 4: 結果描画 ---
    print(f"Candidates found: {len(candidates)}")
    
    if len(candidates) == 0:
        # 何も見つからなかった場合、一番「複雑な形」を無理やりマークする（ネタ用）
        print("No strict candidates. Force detection mode.")
        if len(contours) > 0:
             # 面積がある程度大きいものの中からランダムに...ではなく面積最大のものなどを選ぶ
             cnt = max(contours, key=cv2.contourArea)
             (x,y), radius = cv2.minEnclosingCircle(cnt)
             candidates.append({'x': int(x), 'y': int(y), 'r': int(radius), 'score': 0})

    for c in candidates:
        # 赤丸描画
        cv2.circle(output_img, (c['x'], c['y']), c['r'] + 10, (0, 0, 255), 5)
        # スコア（凹みの数）を表示
        cv2.putText(output_img, f"Lv.{c['score']}", (c['x'] - 20, c['y']), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # 保存
    filename, ext = os.path.splitext(input_path)
    output_path = f"clover_marked.png"
    cv2.imwrite(output_path, output_img)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
