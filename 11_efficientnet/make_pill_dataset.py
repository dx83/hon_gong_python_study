from pathlib import Path
import shutil

# 원본 이미지 폴더
src_dir = Path(r"C:\Workspace\git\hon_gong_python_study\11_efficientnet\drug_img_cut")

# 결과 폴더
dst_root = Path(r"C:\Workspace\git\hon_gong_python_study\11_efficientnet\pill_datasets")

# 결과 폴더 생성
dst_root.mkdir(exist_ok=True)

count = 0

for file_path in src_dir.iterdir():

    if not file_path.is_file():
        continue

    # png, jpg, jpeg만 처리
    if file_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
        continue

    filename = file_path.stem

    # 첫 번째 _ 와 두 번째 _ 사이까지 추출
    parts = filename.split("_")

    if len(parts) < 3:
        print(f"건너뜀: {file_path.name}")
        continue

    pill_name = f"{parts[0]}_{parts[1]}"

    # 알약 폴더 생성
    pill_dir = dst_root / pill_name
    pill_dir.mkdir(exist_ok=True)

    # 파일 복사
    shutil.copy2(
        file_path,
        pill_dir / file_path.name
    )

    count += 1

print(f"완료: {count}개 파일 처리")
print(f"저장 위치: {dst_root}")