import requests
import csv
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://ipm.ucanr.edu"
LIST_URL = f"{BASE_URL}/PMG/diseases/diseaseslist.html"
OUTPUT_FILE = "diseases.csv"


def fetch_page_text(url):
    """Fetch a disease detail page and return its text content."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script/style tags
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()

        body = soup.find("body")
        if body:
            text = body.get_text(separator=" ", strip=True)
        else:
            text = soup.get_text(separator=" ", strip=True)

        # Collapse whitespace
        return " ".join(text.split())
    except Exception as e:
        print(f"  ERROR fetching {url}: {e}")
        return ""


def main():
    print("Fetching disease list...")
    resp = requests.get(LIST_URL, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table")
    rows = table.find_all("tr")[1:]  # skip header

    # Parse all rows
    entries = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 4:
            continue
        host = cols[0].get_text(strip=True)
        link_tag = cols[1].find("a")
        href = link_tag["href"] if link_tag else None
        scientific_name = cols[2].get_text(strip=True)
        disease_type = cols[3].get_text(strip=True)
        entries.append((host, href, scientific_name, disease_type))

    print(f"Found {len(entries)} rows, fetching detail pages...")
    print(entries[:5])

    # Cache detail page text by URL to avoid re-fetching duplicates
    text_cache = {}
    unique_hrefs = set(e[1] for e in entries if e[1])
    print(f"Unique detail pages to fetch: {len(unique_hrefs)}")

    for i, href in enumerate(unique_hrefs, 1):
        full_url = urljoin(BASE_URL, href)
        print(f"  [{i}/{len(unique_hrefs)}] {full_url}")
        text_cache[href] = fetch_page_text(full_url)
        time.sleep(0.3)  # be polite

    # Write CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["host", "text_detail", "scientific_name", "type"])
        for host, href, scientific_name, disease_type in entries:
            text_detail = text_cache.get(href, "") if href else ""
            writer.writerow([host, text_detail, scientific_name, disease_type])

    # Write JSON with citation field
    import json
    json_output = []
    for host, href, scientific_name, disease_type in entries:
        text_detail = text_cache.get(href, "") if href else ""
        full_url = urljoin(BASE_URL, href) if href else ""
        json_output.append({
            "host": host,
            "text_detail": text_detail,
            "scientific_name": scientific_name,
            "type": disease_type,
            "citation": [full_url] if full_url else [],
        })

    with open("diseases.json", "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)

    print(f"Done! Wrote {len(entries)} rows to {OUTPUT_FILE} and diseases.json")


if __name__ == "__main__":
    main()
