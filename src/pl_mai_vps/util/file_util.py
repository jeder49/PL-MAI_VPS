import csv
import json
import os
import zipfile
from io import BytesIO, StringIO
from typing import Any

import requests
from pathlib import Path
from tqdm import tqdm
import time
import hashlib


def download_with_resume(url, file_path, chunk_size=8192, max_retries=3):
    """
    Download file with progress bar and resume capability
    """
    file_path = Path(file_path)
    temp_path = file_path.with_suffix(file_path.suffix + '.tmp')

    # Get existing file size for resume
    resume_byte_pos = temp_path.stat().st_size if temp_path.exists() else 0

    for attempt in range(max_retries + 1):
        try:
            # Set up headers for resume
            headers = {'Range': f'bytes={resume_byte_pos}-'} if resume_byte_pos else {}

            response = requests.get(url, headers=headers, stream=True, timeout=30)

            # Handle resume responses
            if resume_byte_pos and response.status_code == 206:
                print(f"Resuming download from byte {resume_byte_pos}")
            elif resume_byte_pos and response.status_code == 200:
                print("Server doesn't support resume, starting over")
                resume_byte_pos = 0
                temp_path.unlink(missing_ok=True)
            elif response.status_code not in [200, 206]:
                response.raise_for_status()

            # Get total file size
            if 'content-range' in response.headers:
                total_size = int(response.headers['content-range'].split('/')[-1])
            else:
                total_size = int(response.headers.get('content-length', 0)) + resume_byte_pos

            # Set up progress bar
            progress = tqdm(
                total=total_size,
                initial=resume_byte_pos,
                unit='B',
                unit_scale=True,
                desc=file_path.name
            )

            # Download with resume
            mode = 'ab' if resume_byte_pos else 'wb'
            with open(temp_path, mode) as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
                        progress.update(len(chunk))

            progress.close()

            # Move temp file to final location
            temp_path.rename(file_path)
            print(f"Download completed: {file_path}")
            return file_path

        except (requests.RequestException, IOError) as e:
            print(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries:
                wait_time = 2 ** attempt  # exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                # Update resume position for next attempt
                resume_byte_pos = temp_path.stat().st_size if temp_path.exists() else 0
            else:
                print("Max retries exceeded")
                raise


def download_file_robust(url, file_path):
    """Main download function with existence check"""
    file_path = Path(file_path)

    if file_path.exists():
        print(f"File {file_path.name} already exists, skipping download")
        return file_path

    print(f"Downloading {file_path.name} from {url}")
    return download_with_resume(url, file_path)


def verify_checksum(file_path, expected_hash, algorithm='sha256'):
    """Verify file integrity with checksum"""
    hash_func = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)

    actual_hash = hash_func.hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"Checksum mismatch: expected {expected_hash}, got {actual_hash}")
    return True


def download_with_verification(url, file_path, expected_hash=None):
    """Download with optional checksum verification"""
    file_path = download_file_robust(url, file_path)

    if expected_hash:
        print("Verifying file integrity...")
        verify_checksum(file_path, expected_hash)
        print("Checksum verified!")

    return file_path


def process_zip_from_url(url: str, files_to_extract: list[str]) -> dict[str, bytes]:
    """Download zip to RAM and extract specific files"""

    print("Downloading files", files_to_extract, "from zip from", url)
    response = requests.get(url)
    response.raise_for_status()

    # Create a BytesIO object from the response content
    zip_buffer = BytesIO(response.content)

    extracted_files = {}
    with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
        for filename in files_to_extract:
            assert filename in zip_file.namelist(), f"Could not find file {filename} in downloaded zip file!"

            # Read file content directly into memory
            file_content = zip_file.read(filename)
            extracted_files[filename] = file_content
            # Or if it's text: extracted_files[filename] = file_content.decode('utf-8')

    print("Download completed")
    return extracted_files


def csv_bytes_to_dict(csv_bytes: bytes, key_header: str) -> Any:
    """
    Convert CSV bytes to a dictionary with specified column as key.

    Args:
        csv_bytes (bytes): CSV data in bytes format
        key_header (str): Column name to use as dictionary keys

    Returns:
        dict: Dictionary with key_header values as keys and row data as values
    """
    # Convert bytes to string
    csv_string = csv_bytes.decode('utf-8')

    # Create a StringIO object to treat string as file-like object
    csv_file = StringIO(csv_string)

    # Parse CSV
    reader = csv.DictReader(csv_file)

    # Convert to dictionary with specified key
    result = {}
    for row in reader:
        if key_header in row:
            key = row[key_header]
            del row[key_header]
            result[key] = row
        else:
            raise ValueError(f"Key header '{key_header}' not found in CSV columns ({",".join(row.keys())})")

    return result


def json_to_csv_in_memory(url):
    """Download JSON and convert to CSV in memory"""
    response = requests.get(url)
    response.raise_for_status()

    # Parse JSON directly from response
    data = response.json()  # or json.loads(response.text)

    # Method 1: Using csv module
    csv_buffer = StringIO()
    if isinstance(data, list) and len(data) > 0:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    csv_content = csv_buffer.getvalue()
    return csv_content


def save_data_to_file(content: str | Any, file_path: Path):
    if isinstance(content, str):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    else:  # bytes
        with open(file_path, 'wb') as f:
            f.write(content)

    return file_path


def read_json_file(file_path: Path) -> Any:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_directory_from_zip(zip_path: Path, target_dir_in_zip: str, extract_to_path: Path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get all file names in the zip
        all_files = zip_ref.namelist()

        # Filter files that are in the target directory and are .mp4 files
        target_files = [
            file for file in all_files
            if file.startswith(target_dir_in_zip) and file.endswith('.mp4')
        ]

        # Extract each target file
        for file in tqdm(target_files, desc="Extracting videos"):
            # Get just the filename without the directory path
            filename = os.path.basename(file)

            # Read the file data from zip
            file_data = zip_ref.read(file)

            # Write it to the target directory with just the filename
            output_path = extract_to_path / filename
            with open(output_path, 'wb') as output_file:
                output_file.write(file_data)
