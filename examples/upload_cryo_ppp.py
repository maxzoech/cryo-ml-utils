import os
import tarfile
import requests
import boto3
import tempfile
from tqdm.auto import tqdm
from io import BytesIO
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def fetch_files(url):

    # Fetch the webpage
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all links on the page
    for link in soup.find_all("a", href=True):
        file_url = urljoin(url, link["href"])

        # Skip non-file links (e.g., directories or other pages)
        if not any(file_url.lower().endswith(ext) for ext in [".tar.gz"]):
            continue

        if file_url.lower().endswith("10005.tar.gz"):
            print("Skipping 10005 as it is already in S3")
            continue

        yield file_url


def download_tar_gz(url, local_path):
    """Download a .tar.gz file from a URL with a progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Get the total file size from the response headers
    total_size = int(response.headers.get("content-length", 0))

    # Initialize the progress bar
    progress_bar = tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        leave=False,
        desc=f"Downloading {url}",
    )

    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            # Update the progress bar
            progress_bar.update(len(chunk))

    progress_bar.close()


def extract_tar_gz(tar_path, extract_to):
    """Extract a .tar.gz file."""
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_to)


def upload_to_s3(s3, *, local_dir, bucket):
    def _files_gen(local_dir, s3_prefix=""):
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                s3_key = os.path.join(s3_prefix, os.path.relpath(local_path, local_dir))
                yield local_path, s3_key

    files = list(_files_gen(local_dir, s3_prefix="cryoPPP"))
    for local_path, s3_key in tqdm(files, desc="Uploading to S3"):
        s3.upload_file(local_path, bucket, s3_key)


def main():
    errors = []

    s3_client = boto3.client(
        "s3",
        region_name="eu-north-1",
    )

    for ds_url in tqdm(fetch_files("https://calla.rnet.missouri.edu/cryoppp/")):
        with tempfile.TemporaryDirectory() as temp_dir:
            compressed_file = os.path.join(temp_dir, "downloaded.tar.gz")
            uncompressed_dir = os.path.join(temp_dir, "downloaded")

            try:
                download_tar_gz(ds_url, compressed_file)
                extract_tar_gz(compressed_file, uncompressed_dir)
                upload_to_s3(s3_client, local_dir=uncompressed_dir, bucket="s3-ucm")

            except Exception as e:
                errors.append((ds_url, e))

    print(f"{len(errors)} errors occurred")
    for url, error in errors:
        print(f"{url}: {str(error)}")


if __name__ == "__main__":
    main()
