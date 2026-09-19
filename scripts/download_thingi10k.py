import os
import sys
import time
import tarfile
import httpx

snapshot_dir = r"E:\Datasets\huggingface\hub\datasets--Thingi10K--Thingi10K\snapshots\2d5d3b2f3cd3711028ad75b12788c13b25559ec6"
os.makedirs(snapshot_dir, exist_ok=True)

tar_target = os.path.join(snapshot_dir, "Thingi10K_npz.tar.gz")
part_file = tar_target + ".part"

url = "https://huggingface.co/datasets/Thingi10K/Thingi10K/resolve/v1.5.0/Thingi10K_npz.tar.gz"

if os.path.exists(tar_target):
    size_mb = os.path.getsize(tar_target) / (1024 * 1024)
    print(f"Target file already exists: {tar_target} ({size_mb:.2f} MB)")
    if size_mb > 3800:
        print("Verifying existing tar archive...")
        if tarfile.is_tarfile(tar_target):
            print("SUCCESS: Existing Thingi10K_npz.tar.gz is complete and valid!")
            sys.exit(0)
        else:
            print("Existing archive is corrupt, will re-download.")

# Step 1: Query Hugging Face with retry to resolve CDN download endpoint
print("Resolving CDN download endpoint via Hugging Face...")
cdn_url = None
for attempt in range(1, 10):
    try:
        with httpx.Client(verify=False, follow_redirects=False, timeout=20.0) as c1:
            res = c1.get(url)
            if res.status_code in (301, 302, 307, 308) and "location" in res.headers:
                cdn_url = res.headers["location"]
                print(f"Resolved CDN endpoint (attempt {attempt}): {cdn_url.split('?')[0]}")
                break
            elif res.status_code == 200:
                cdn_url = str(res.url)
                print(f"Direct endpoint (attempt {attempt})")
                break
    except Exception as exc:
        print(f"Attempt {attempt}/10 resolving link failed ({type(exc).__name__}), retrying in 2s...")
        time.sleep(2.0)

if not cdn_url:
    print("ERROR: Could not resolve CDN download link after 10 attempts.")
    sys.exit(1)

# Step 2: Stream download with auto-resume from AWS CDN directly
chunk_size = 2 * 1024 * 1024  # 2 MB chunks
max_retries = 20

for attempt in range(1, max_retries + 1):
    existing_bytes = os.path.getsize(part_file) if os.path.exists(part_file) else 0
    headers = {}
    if existing_bytes > 0:
        headers["range"] = f"bytes={existing_bytes}-"
        print(f"Resuming download from {existing_bytes / 1024 / 1024:.2f} MB...")
    else:
        print("Starting download of Thingi10K_npz.tar.gz (~4.08 GB)...")

    try:
        with httpx.Client(verify=False, trust_env=False, timeout=60.0) as c2:
            with c2.stream("GET", cdn_url, headers=headers) as response:
                response.raise_for_status()

                if response.status_code == 206:
                    total_bytes = existing_bytes + int(response.headers.get("content-length", 0))
                    mode = "ab"
                elif response.status_code == 200:
                    total_bytes = int(response.headers.get("content-length", 0))
                    existing_bytes = 0
                    mode = "wb"
                else:
                    print(f"Unexpected status code: {response.status_code}")
                    break

                total_mb = total_bytes / (1024 * 1024)
                print(f"Total target size: {total_mb:.2f} MB")

                downloaded = existing_bytes
                last_report_time = time.time()
                last_report_bytes = downloaded

                with open(part_file, mode) as f:
                    for chunk in response.iter_bytes(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last_report_time >= 5.0 or downloaded >= total_bytes:
                            speed_mb = (downloaded - last_report_bytes) / (1024 * 1024) / max(now - last_report_time, 0.1)
                            pct = (downloaded / total_bytes * 100.0) if total_bytes > 0 else 0.0
                            remaining_mb = (total_bytes - downloaded) / (1024 * 1024)
                            eta_s = (remaining_mb / speed_mb) if speed_mb > 0.01 else 9999
                            print(f"[{pct:5.1f}%] {downloaded / 1024 / 1024:7.1f} / {total_mb:.1f} MB | Speed: {speed_mb:5.2f} MB/s | ETA: {eta_s / 60:4.1f} min", flush=True)
                            last_report_time = now
                            last_report_bytes = downloaded

                if downloaded >= total_bytes:
                    print("\nDownload complete! Finalizing file...")
                    if os.path.exists(tar_target):
                        os.remove(tar_target)
                    os.rename(part_file, tar_target)
                    print(f"Verifying {tar_target} integrity...")
                    if tarfile.is_tarfile(tar_target):
                        print(f"SUCCESS: Thingi10K_npz.tar.gz verified successfully! Saved to {tar_target}")
                    else:
                        print("WARNING: File downloaded but tarfile check failed.")
                    sys.exit(0)
    except Exception as exc:
        print(f"Stream interrupted ({type(exc).__name__}: {exc}), will auto-resume in 3s (attempt {attempt}/{max_retries})...")
        time.sleep(3.0)

print("Failed to complete download after max retries.")
sys.exit(1)
