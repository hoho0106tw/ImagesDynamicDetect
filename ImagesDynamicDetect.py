import os
from pathlib import Path
import numpy as np
import pydicom
import cv2
import matplotlib.pyplot as plt

def load_dicom_series(input_path: str) -> np.ndarray:
    """讀取單一 DICOM 檔(多影格) 或資料夾內多個 DICOM，輸出 shape=(T,H,W)"""
    path = Path(input_path)
    if path.is_file():
        ds = pydicom.dcmread(str(path))
        if hasattr(ds, "NumberOfFrames") and ds.NumberOfFrames:
            arr = ds.pixel_array  # (T,H,W)
        else:
            arr = ds.pixel_array[np.newaxis, ...]
    else:
        files = sorted(path.glob("*.dcm"))
        arrs = [pydicom.dcmread(str(f)).pixel_array for f in files]
        arr = np.stack(arrs, axis=0)
    return arr.astype(np.float32)

def detect_motion_binary(stack: np.ndarray, threshold: int = 30, min_area: int = 50):
    motions = []
    masks = []
    for i in range(1, stack.shape[0]):
        prev = stack[i-1]
        curr = stack[i]

        # --- Step 1: 差異 ---
        diff = cv2.absdiff(curr, prev)

        # --- Step 2: 正規化到 0-255 並轉成 uint8 ---
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        diff_u8 = diff_norm.astype(np.uint8)

        # --- Step 3: 二值化 ---
        _, binary = cv2.threshold(diff_u8, threshold, 255, cv2.THRESH_BINARY)

        # --- Step 4: 確保單通道 uint8 ---
        if len(binary.shape) > 2:  # 如果是三通道
            binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
        binary = np.ascontiguousarray(binary, dtype=np.uint8)

        print("DEBUG:", binary.dtype, binary.shape, binary.min(), binary.max())  # ← 應該印出 uint8 (H,W) 0 255

        # --- Step 5: 找輪廓 ---
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = any(cv2.contourArea(c) > min_area for c in contours)

        motions.append(motion_detected)
        masks.append(binary)
    return motions, masks


def detect_motion_binary_once(stack: np.ndarray, threshold: int = 30, min_area: int = 50) -> bool:
    """
    偵測 DICOM 序列是否有移動
    - True: 只要有一幀符合移動條件
    - False: 全部幀都沒有移動
    """
    for i in range(1, stack.shape[0]):
        prev = stack[i-1]
        curr = stack[i]

        # --- Step 1: 差異 ---
        diff = cv2.absdiff(curr, prev)

        # --- Step 2: 正規化到 0-255 並轉成 uint8 ---
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        diff_u8 = diff_norm.astype(np.uint8)

        # --- Step 3: 二值化 ---
        _, binary = cv2.threshold(diff_u8, threshold, 255, cv2.THRESH_BINARY)

        # --- Step 4: 確保單通道 uint8 ---
        if len(binary.shape) > 2:
            binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
        binary = np.ascontiguousarray(binary, dtype=np.uint8)

        # --- Step 5: 找輪廓 ---
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = any(cv2.contourArea(c) > min_area for c in contours)

        if motion_detected:  # ✅ 一偵測到就直接回傳
            return True

    return False  # 全部檢查完都沒有移動





if __name__ == "__main__":
    input_path = r"C:\Users\Administrator\Downloads\CVUS (2)\LVEF大於40\1448991\S812904525820010\1.2.528.1.1001.200.10.1829.6765.1350351824.20250715003620287\SDY00000\SRS00000\IMG00015.DCM"
    stack = load_dicom_series(input_path)

  

    has_motion = detect_motion_binary_once(stack, threshold=30, min_area=50)

    if has_motion:
        print("⚠ 偵測到移動影像")
    else:
        print("✅ 沒有偵測到移動")
