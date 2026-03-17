import cv2,sys,os,numpy as np
def main():
    if len(sys.argv)<2: return print("Usage: exe <img_path>")
    if not os.path.exists(sys.argv[1]): return print("File not found")
    img=cv2.imread(sys.argv[1])
    if img is None: return print("Image read error")
    out=img.copy(); hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array([35,40,40]),np.array([85,255,255]))
    kernel=np.ones((5,5),np.uint8)
    mask=cv2.morphologyEx(cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel),cv2.MORPH_OPEN,kernel)
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c)<1000: continue
        hull=cv2.convexHull(c,returnPoints=False)
        if hull is None or len(hull)<=3: continue
        defs=cv2.convexityDefects(c,hull)
        if defs is None: continue
        valid_defs=0
        for i in range(defs.shape[0]):
            s,e,f,d=defs[i,0]
            if d>2000: valid_defs+=1
        if valid_defs==4:
            M=cv2.moments(c)
            if M["m00"]!=0:
                cx,cy=int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"])
                cv2.circle(out,(cx,cy),50,(0,0,255),4); cv2.putText(out,"4-LEAF!",(cx-30,cy-60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                print(f"CLOVER at ({cx}, {cy})")
    cv2.imwrite("clover_marked.png",out); print("Saved.")
if __name__=="__main__": main()
