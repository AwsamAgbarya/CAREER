import requests
import os
import time
import sys

# Import the access key from secrets.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from configs.secrets import UNSPLASH_ACCESS_KEY
except ImportError:
    UNSPLASH_ACCESS_KEY = None
    print("WARNING: configs/secrets.py not found. Please create it with your API key.")


def download_unsplash_images(keywords, num_images=100, output_dir="./backgrounds/unsplash"):
    """
    Download images from Unsplash API filtered by keywords
    
    Args:
        keywords: List of search keywords (e.g., ["laboratory", "workspace", "lab bench"])
        num_images: Total number of images to download
        output_dir: Directory to save images
    """
    
    ACCESS_KEY = UNSPLASH_ACCESS_KEY
    
    if ACCESS_KEY is None:
        print("ERROR: Please get your free API key from https://unsplash.com/developers")
        print("1. Create an account")
        print("2. Go to 'Your apps' and create a new application")
        print("3. Copy your 'Access Key'")
        print("4. Create a file named 'secrets.py' with:")
        print("   UNSPLASH_ACCESS_KEY = 'your_key_here'")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    images_per_keyword = num_images // len(keywords)
    downloaded = 0
    
    print(f"Downloading {num_images} images for keywords: {keywords}")
    
    for keyword in keywords:
        print(f"\nSearching for '{keyword}'...")
        
        # Unsplash API endpoint for random photos
        url = "https://api.unsplash.com/photos/random"
        
        # Request multiple images at once (max 30 per request)
        batch_size = min(30, images_per_keyword)
        batches = (images_per_keyword + batch_size - 1) // batch_size
        
        for batch in range(batches):
            remaining = images_per_keyword - (batch * batch_size)
            count = min(batch_size, remaining)
            
            params = {
                'query': keyword,
                'client_id': ACCESS_KEY,
                'count': count,
                'orientation': 'landscape'  # Optional: get landscape images
            }
            
            try:
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    photos = response.json()
                    
                    for photo in photos:
                        # Get the regular size image URL
                        img_url = photo['urls']['regular']
                        photo_id = photo['id']
                        
                        # Download the image
                        img_response = requests.get(img_url)
                        
                        if img_response.status_code == 200:
                            filename = f"{keyword.replace(' ', '_')}_{photo_id}.jpg"
                            filepath = os.path.join(output_dir, filename)
                            
                            with open(filepath, 'wb') as f:
                                f.write(img_response.content)
                            
                            downloaded += 1
                            print(f"  [{downloaded}/{num_images}] Downloaded: {filename}")
                        
                        # Rate limiting - be nice to the API
                        time.sleep(0.5)
                
                elif response.status_code == 403:
                    print(f"  Error: Rate limit exceeded. Wait a bit and try again.")
                    break
                else:
                    print(f"  Error: API returned status {response.status_code}")
                    
            except Exception as e:
                print(f"  Error downloading: {e}")
        
        # Rate limiting between keywords
        time.sleep(1)
    
    print(f"\nSuccessfully downloaded {downloaded} images to {output_dir}")


if __name__ == "__main__":
    # Customize your keywords here
    keywords = [
        "laboratory",
        "lab bench",
        "workspace",
        "office desk",
        "workbench"
    ]
    
    download_unsplash_images(
        keywords=keywords,
        num_images=40,
        output_dir="./backgrounds/environments/"
    )
