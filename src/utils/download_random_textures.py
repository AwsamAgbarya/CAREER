import requests
import random
import os
import re

def download_random_ambientcg_textures(num_textures=100, output_dir="./backgrounds"):
    """Download random flat texture images from ambientCG"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all available textures
    api_url = "https://ambientcg.com/api/v2/full_json"
    params = {
        "type": "Material",
        "limit": 999,
    }
    
    print("Fetching texture list from ambientCG...")
    response = requests.get(api_url, params=params)
    data = response.json()
    
    all_assets = data.get("foundAssets", [])
    print(f"Found {len(all_assets)} total textures")
    
    # Randomly sample
    selected = random.sample(all_assets, min(num_textures, len(all_assets)))
    
    downloaded = 0
    for i, asset in enumerate(selected):
        asset_id = asset.get("assetId", "unknown")
        
        # Extract color map URL from previewLinks
        color_url = None
        preview_links = asset.get("previewLinks", [])
        
        if preview_links and len(preview_links) > 0:
            # Get the first preview link which contains all the texture URLs
            first_link = preview_links[0]
            link_url = first_link.get("url", "")
            
            # Parse out the color_url parameter from the URL
            color_match = re.search(r'color_url=(https://[^&]+)', link_url)
            if color_match:
                color_url = color_match.group(1)
        
        if color_url:
            try:
                print(f"[{i+1}/{num_textures}] Downloading {asset_id}...")
                
                # Download the color texture
                img_response = requests.get(color_url, timeout=30)
                img_response.raise_for_status()
                
                # Determine extension from URL
                ext = ".jpg" if ".jpg" in color_url.lower() else ".png"
                save_path = os.path.join(output_dir, f"{asset_id}{ext}")
                
                with open(save_path, "wb") as f:
                    f.write(img_response.content)
                
                downloaded += 1
                print(f"  ✓ Downloaded")
                        
            except Exception as e:
                print(f"  ✗ Failed: {e}")
        else:
            print(f"[{i+1}/{num_textures}] Skipping {asset_id} (no color URL found)")
    
    print(f"\nSuccessfully downloaded {downloaded}/{num_textures} flat textures to {output_dir}")

# Download 100 random textures
output_dir = "./backgrounds/textures"
os.makedirs(output_dir, exist_ok=True)
download_random_ambientcg_textures(100, output_dir)
