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
from PIL import Image

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


def find_pills(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, w = image.shape[:2]

    corners = np.concatenate([
        hsv[:50, :50].reshape(-1, 3),
        hsv[:50, -50:].reshape(-1, 3),
        hsv[-50:, :50].reshape(-1, 3),
        hsv[-50:, -50:].reshape(-1, 3)
    ])

    bg_color = np.median(corners, axis=0)

    diff = np.linalg.norm(
        hsv.astype(np.float32) - bg_color,
        axis=2
    )

    mask = (diff > 20).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    candidates = []

    img_area = h * w

    for i in range(1, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w_box = stats[i, cv2.CC_STAT_WIDTH]
        h_box = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 1000:
            continue

        if area > img_area * 0.5:
            continue

        candidates.append(
            (area, x, y, w_box, h_box)
        )

    candidates.sort(reverse=True)

    return candidates[:4]


def crop_and_save(image, candidates, output_folder, base_name):

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

        save_path = output_folder / f"{base_name}_{saved+1}.png"

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

                print(f"[저장 실패] {save_path}")

        except Exception as e:

            print(f"[저장 오류] {save_path}")
            print(e)

        if saved >= MAX_OBJECTS:
            break

    return saved


def process_image(image_path):

    try:

        #image = cv2.imread(str(image_path))

        #if image is None:
        #    print(f"[실패] 읽기 오류 : {image_path}")
        #    return

        pil_img = Image.open(image_path)

        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        image = cv2.cvtColor(
            np.array(pil_img),
            cv2.COLOR_RGB2BGR
        )

        base_name = extract_base_name(image_path.name)

        output_folder = OUTPUT_DIR / base_name
        output_folder.mkdir(exist_ok=True)

        candidates = find_pills(image)

        if len(candidates) == 0:
            print(f"[검출 실패] {image_path.name}")
            return

        saved_count = crop_and_save(
            image,
            candidates,
            output_folder,
            base_name
        )

        print(
            f"[완료] {image_path.name} -> {saved_count}개 저장"
        )

    except Exception as e:
        #print(f"[오류] {image_path.name}")
        print(f"[실패] 읽기 오류 : {image_path}")
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
