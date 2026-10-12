import base64

with open('data/oct_gbdt_chart.png', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode('utf-8')

html = f"""<!DOCTYPE html>
<html>
<head>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-2">
  <div class="bg-[var(--card)] text-[var(--foreground)] border border-[var(--border)] rounded-xl p-2 shadow-sm flex flex-col items-center">
    <img src="data:image/png;base64,{b64}" class="w-full h-auto rounded" alt="Chart" />
  </div>
</body>
</html>"""

with open(r'C:\Users\chabo\.gemini\antigravity\brain\a3a2ce53-0b98-4007-b06e-e7780dc6e52e\chart_widget.html', 'w', encoding='utf-8') as out:
    out.write(html)
print("Widget created successfully.")
