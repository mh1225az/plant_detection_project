import os
import requests
from dotenv import load_dotenv
from googleapiclient.discovery import build
from PIL import Image
from io import BytesIO

# Load API Key dan CSE ID dari .env
load_dotenv()
API_KEY = os.getenv("API_KEY")
CSE_ID = os.getenv("CSE_ID")

def is_valid_image(content):
    try:
        img = Image.open(BytesIO(content))
        img.verify()
        return True
    except Exception:
        return False

def download_images(query, save_dir, num_images=50):
    service = build("customsearch", "v1", developerKey=API_KEY)
    os.makedirs(save_dir, exist_ok=True)

    start = 1
    downloaded = 0

    while downloaded < num_images:
        res = service.cse().list(
            q=query,
            cx=CSE_ID,
            searchType="image",
            num=10,
            start=start,
            imgType="photo",
            imgSize="MEDIUM",       # ← Huruf kapital semua!
            safe="high"              # Filter konten tidak pantas
        ).execute()

        if "items" not in res:
            print("[!] Tidak ada hasil lagi.")
            break

        for item in res.get("items", []):
            try:
                img_url = item["link"]
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                }
                response = requests.get(img_url, headers=headers, timeout=10)

                # Validasi tipe gambar
                if response.status_code == 200 and is_valid_image(response.content):
                    file_path = os.path.join(save_dir, f"{query.replace(' ', '_')}_{downloaded+1}.jpg")
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    downloaded += 1
                    print(f"[✓] {query} - {downloaded}/{num_images}")
                else:
                    print(f"[!] Gambar tidak valid, dilewati.")
                
                if downloaded >= num_images:
                    break
            except Exception as e:
                print(f"[!] Gagal download: {e}")
        start += 10

# Daftar tanaman
plant_list = [
    # 'Gejala Antraknosa (Patek) pada tanaman cabai',
    # 'Gejala Gemini Virus pada tanaman cabai',
    'Gejala Penyakit bercak daun pada tanaman cabai'
    # 'Gejala Penyakit Layu Fusarium pada tanaman cabai'
]

for plant in plant_list:
    download_images(plant, f'dataset/{plant}', num_images=50)
