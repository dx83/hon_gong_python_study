# -*- coding: utf-8 -*-
"""
알약 이미지 자동 크롭 스크립트

실행 위치
---------
project_root/
│
├─ pills_crop_script.py   ← 이 파일
├─ images/
│   ├─ 01/
│   ├─ 02/
│   ├─ ...
│   └─ 28/
│
└─ images_crop/    ← 자동 생성

실행
----
python crop_pills.py

필요 패키지
-----------
pip install opencv-python

동작
----
1. images 하위 모든 jpg 탐색
2. OpenCV Contour 검출
3. 면적 기준 알약 후보 선택
4. 알약별 Bounding Box Crop
5. Padding 추가
6. PNG 저장

출력 예시
---------
images_crop/
└─ 200101951_한신타미놀캅셀(아세트아미노펜)/
   ├─ 200101951_한신타미놀캅셀(아세트아미노펜)_1.png
   └─ 200101951_한신타미놀캅셀(아세트아미노펜)_2.png
"""

import os
import cv2
import numpy as np
from pathlib import Path

# ==========================
# 설정값
# ==========================

PADDING_RATIO = 0.20      # 20% 여백
MIN_AREA = 3000           # 최소 객체 면적
MAX_OBJECTS = 10          # 최대 저장 개수

# ==========================
# 경로
# ==========================

ROOT_DIR = Path.cwd()

IMAGES_DIR = ROOT_DIR / "images"
OUTPUT_DIR = ROOT_DIR / "images_crop"

OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================
# 함수
# ==========================

def extract_base_name(filename: str) -> str:
    """
    파일명:
    200101951_한신타미놀캅셀(아세트아미노펜)_1pw5pyj7qmnxr01.jpg

    반환:
    200101951_한신타미놀캅셀(아세트아미노펜)
    """

    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"

    return stem


def find_pill_HSV(image):
    h_img, w_img = image.shape[:2]

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # 모서리 색상으로 배경 추정
    corner_size = 50

    corners = np.concatenate([
        hsv[:corner_size, :corner_size].reshape(-1, 3),
        hsv[:corner_size, -corner_size:].reshape(-1, 3),
        hsv[-corner_size:, :corner_size].reshape(-1, 3),
        hsv[-corner_size:, -corner_size:].reshape(-1, 3)
    ])

    bg_color = np.median(
        corners,
        axis=0
    )

    # 배경과의 거리 계산
    diff = np.linalg.norm(
        hsv.astype(np.float32) -
        bg_color.astype(np.float32),
        axis=2
    )

    # 배경 제거
    mask = np.zeros_like(diff, dtype=np.uint8)
    mask[diff > 20] = 255

    # 노이즈 제거
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    img_area = h_img * w_img

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < MIN_AREA:
            continue

        if area > img_area * 0.5:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        ratio = w / float(h)

        if ratio < 0.2 or ratio > 8.0:
            continue

        candidates.append(
            (area, x, y, w, h)
        )

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return candidates


def crop_and_save(image, candidates, base_name):

    h_img, w_img = image.shape[:2]

    saved = 0

    for _, x, y, w, h in candidates:

        pad_x = int(w * PADDING_RATIO)
        pad_y = int(h * PADDING_RATIO)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)

        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        save_path = (
            OUTPUT_DIR /
            f"{base_name}_{saved+1}.png"
        )

        try:

            result, encoded = cv2.imencode(
                ".png",
                crop,
                [cv2.IMWRITE_PNG_COMPRESSION, 3]
            )

            if result:

                encoded.tofile(str(save_path))

                saved += 1

            else:

                print(
                    f"[저장 실패] {save_path}"
                )

        except Exception as e:

            print(
                f"[저장 오류] {save_path}"
            )

            print(e)

    return saved


def read_image_unicode(image_path):

    try:

        data = np.fromfile(
            str(image_path),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            data,
            cv2.IMREAD_COLOR
        )

        return image

    except Exception:

        return None
    

def process_image(image_path):

    try:

        image = read_image_unicode(image_path)

        if image is None:
            print(f"[실패] 읽기 오류 : {image_path}")
            return

        base_name = extract_base_name(image_path.name)

        candidates = find_pill_HSV(image)

        # 알약 2개가 아니면 제외
        if len(candidates) != 2:

            print(
                f"[제외] {image_path.name} "
                f"(검출 {len(candidates)}개)"
            )

            return

        saved_count = crop_and_save(
            image,
            candidates,
            base_name
        )

        if saved_count != 2:

            # 저장 실패 시 삭제
            for i in range(1, saved_count + 1):

                png_path = (
                    OUTPUT_DIR /
                    f"{base_name}_{i}.png"
                )

                if png_path.exists():
                    png_path.unlink()

            print(
                f"[제외] {image_path.name} "
                f"(저장 {saved_count}개)"
            )

            return

        print(
            f"[완료] {image_path.name}"
        )

    except Exception as e:

        print(f"[오류] {image_path.name}")
        print(e)

# ==========================
# 메인
# ==========================

def main():

    if not IMAGES_DIR.exists():
        print("images 폴더를 찾을 수 없습니다.")
        return

    image_files = []

    for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
        image_files.extend(
            IMAGES_DIR.rglob(ext)
        )

    total = len(image_files)

    print("=" * 60)
    print("알약 이미지 자동 크롭 시작")
    print(f"총 이미지 수 : {total}")
    print("=" * 60)

    for idx, image_path in enumerate(image_files, start=1):

        print(
            f"[{idx}/{total}] 처리중...",
            end=" "
        )

        process_image(image_path)

    print("=" * 60)
    print("모든 작업 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
