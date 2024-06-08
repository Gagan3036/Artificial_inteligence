from bing_image_downloader import downloader

# Function to download images from Bing Images
def download_images(query, limit):
    downloader.download(query, limit=limit, output_dir=query, adult_filter_off=True, force_replace=False, timeout=60)

# Example usage
query = "HarryPotterFace"
limit = 30
download_images(query, limit)
