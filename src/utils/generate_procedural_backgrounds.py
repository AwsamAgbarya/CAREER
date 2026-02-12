import numpy as np
from PIL import Image
import os
from scipy.ndimage import zoom

def generate_perlin_noise(shape, scale=10, octaves=6, persistence=0.5, seed=None):
    """Simplified Perlin-like noise using interpolated random values"""
    if seed is not None:
        np.random.seed(seed)
    
    h, w = shape
    result = np.zeros(shape)
    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0
    
    for _ in range(octaves):
        grid_size = max(2, int(min(h, w) / (scale * frequency)))
        low_res = np.random.rand(grid_size, grid_size)
        
        octave = zoom(low_res, (h / grid_size, w / grid_size), order=1)
        octave = octave[:h, :w]
        
        result += octave * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= 2.0
    
    result = result / max_value
    return result

def generate_gradient(shape, gradient_type='linear', angle=0, colors=None, seed=None):
    """Generate color gradient"""
    if seed is not None:
        np.random.seed(seed)
    
    h, w = shape
    
    if colors is None:
        colors = [np.random.rand(3), np.random.rand(3)]
    
    if gradient_type == 'linear':
        angle_rad = np.radians(angle)
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)
        
        gradient = xx * np.cos(angle_rad) + yy * np.sin(angle_rad)
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
        
    elif gradient_type == 'radial':
        y, x = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        gradient = np.sqrt((x - cx)**2 + (y - cy)**2)
        gradient = gradient / gradient.max()
    
    img = np.zeros((h, w, 3))
    for i in range(3):
        img[:, :, i] = colors[0][i] * (1 - gradient) + colors[1][i] * gradient
    
    return img

def generate_voronoi(shape, num_points=50, seed=None):
    """Generate Voronoi/cellular texture"""
    if seed is not None:
        np.random.seed(seed)
    
    h, w = shape
    points = np.random.rand(num_points, 2)
    points[:, 0] *= h
    points[:, 1] *= w
    
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.stack([yy.ravel(), xx.ravel()], axis=1)
    
    min_dist = np.full(h * w, np.inf)
    for point in points:
        dist = np.sqrt((coords[:, 0] - point[0])**2 + (coords[:, 1] - point[1])**2)
        min_dist = np.minimum(min_dist, dist)
    
    result = min_dist.reshape(h, w)
    result = result / result.max()
    
    return result

def generate_checkerboard(shape, square_size=32, seed=None):
    """Generate checkerboard pattern"""
    if seed is not None:
        np.random.seed(seed)
    
    h, w = shape
    y_squares = (np.arange(h) // square_size) % 2
    x_squares = (np.arange(w) // square_size) % 2
    
    checkerboard = (y_squares[:, None] + x_squares[None, :]) % 2
    
    return checkerboard.astype(float)

def generate_gaussian_noise(shape, mean=0.5, std=0.2, seed=None):
    """Generate Gaussian noise"""
    if seed is not None:
        np.random.seed(seed)
    
    noise = np.random.normal(mean, std, shape)
    noise = np.clip(noise, 0, 1)
    
    return noise

def generate_procedural_backgrounds(output_dir, num_each=20, image_size=(1024, 1280)):
    """Generate all types of procedural backgrounds
    
    Args:
        output_dir: Directory to save generated images
        num_each: Number of each type to generate
        image_size: (height, width) of output images
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    h, w = image_size
    count = 0
    
    print(f"Generating procedural backgrounds to {output_dir}...")
    
    # 1. Perlin noise with different colors
    for i in range(num_each):
        noise = generate_perlin_noise((h, w), scale=np.random.randint(20, 80), 
                                     octaves=np.random.randint(4, 8), seed=count)
        
        color1 = np.random.rand(3)
        color2 = np.random.rand(3)
        
        img = np.zeros((h, w, 3))
        for c in range(3):
            img[:, :, c] = color1[c] * (1 - noise) + color2[c] * noise
        
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(output_dir, f"perlin_{i:03d}.jpg"))
        count += 1
    
    print(f"  ✓ Generated {num_each} Perlin noise textures")
    
    # 2. Linear gradients
    for i in range(num_each):
        angle = np.random.randint(0, 360)
        img = generate_gradient((h, w), 'linear', angle=angle, seed=count)
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(output_dir, f"gradient_linear_{i:03d}.jpg"))
        count += 1
    
    print(f"  ✓ Generated {num_each} linear gradients")
    
    # 3. Radial gradients
    for i in range(num_each):
        img = generate_gradient((h, w), 'radial', seed=count)
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(output_dir, f"gradient_radial_{i:03d}.jpg"))
        count += 1
    
    print(f"  ✓ Generated {num_each} radial gradients")
    
    # 4. Voronoi patterns
    for i in range(num_each):
        voronoi = generate_voronoi((h, w), num_points=np.random.randint(20, 100), seed=count)
        
        color1 = np.random.rand(3)
        color2 = np.random.rand(3)
        
        img = np.zeros((h, w, 3))
        for c in range(3):
            img[:, :, c] = color1[c] * (1 - voronoi) + color2[c] * voronoi
        
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(output_dir, f"voronoi_{i:03d}.jpg"))
        count += 1
    
    print(f"  ✓ Generated {num_each} Voronoi patterns")
    
    # 5. Checkerboards
    for i in range(num_each):
        checker = generate_checkerboard((h, w), square_size=np.random.randint(16, 128), seed=count)
        
        color1 = np.random.rand(3)
        color2 = np.random.rand(3)
        
        img = np.zeros((h, w, 3))
        for c in range(3):
            img[:, :, c] = color1[c] * (1 - checker) + color2[c] * checker
        
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(output_dir, f"checkerboard_{i:03d}.jpg"))
        count += 1
    
    print(f"  ✓ Generated {num_each} checkerboards")
    
    # 6. Gaussian noise
    for i in range(num_each):
        noise = generate_gaussian_noise((h, w), mean=np.random.uniform(0.3, 0.7), 
                                       std=np.random.uniform(0.1, 0.3), seed=count)
        
        img = np.stack([noise, noise, noise], axis=2)
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(output_dir, f"gaussian_{i:03d}.jpg"))
        count += 1
    
    print(f"  ✓ Generated {num_each} Gaussian noise patterns")
    
    print(f"\nTotal: {count} procedural backgrounds generated!")
    return count

if __name__ == "__main__":
    generate_procedural_backgrounds(
        output_dir="./backgrounds/procedural",
        num_each=10,  # Generate 20 of each type = 120 total
        image_size=(1024, 1280)  # Match your rendering resolution
    )
