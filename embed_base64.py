import base64
import sys

img_path = r"C:\Users\chabo\.gemini\antigravity\brain\a4ba377f-2239-4e2c-9092-434e9a80c2fd\soxl_comparison.png"
md_path = r"C:\Users\chabo\.gemini\antigravity\brain\a4ba377f-2239-4e2c-9092-434e9a80c2fd\soxl_comparison_report.md"

with open(img_path, "rb") as f:
    b64_str = base64.b64encode(f.read()).decode('utf-8')

data_uri = f"data:image/png;base64,{b64_str}"

with open(md_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace the text line with the base64 image
target = "차트 이미지가 바로 보이지 않으신다면, **[여기를 클릭하여 차트 이미지 열기](file:///C:/Users/chabo/.gemini/antigravity/brain/a4ba377f-2239-4e2c-9092-434e9a80c2fd/soxl_comparison.png)** 를 눌러주세요."
replacement = f"![SOXL 60분봉 ON vs OFF 비교차트]({data_uri})"

content = content.replace(target, replacement)

with open(md_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Artifact updated with base64 image.")
