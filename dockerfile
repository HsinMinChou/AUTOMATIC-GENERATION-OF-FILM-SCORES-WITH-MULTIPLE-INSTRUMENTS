# 1. 基底映像
FROM python:3.9-slim

# 2. 設定工作目錄
WORKDIR /app

# 3. 複製並安裝依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 複製專案程式
COPY . .

# 5. 暴露 Gradio 預設埠
EXPOSE 7860

# 6. 啟動指令
CMD ["python", "app.py"]
