import requests
import io
from PIL import Image

class OmniGenClient:
    def __init__(self, base_url="http://103.219.171.95:8000"):
        self.base_url = base_url.rstrip('/')

    def ping(self):
        """Test the connection to the server"""
        try:
            response = requests.get(f"{self.base_url}/")
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def generate_image(self, prompt, height=1024, width=1024, guidance_scale=2.5, seed=0, save_path=None):
        """Generate an image using the OmniGen model"""
        try:
            payload = {
                "prompt": prompt,
                "height": height,
                "width": width,
                "guidance_scale": guidance_scale,
                "seed": seed
            }
            
            response = requests.post(f"{self.base_url}/generate", json=payload)
            
            if response.status_code == 200:
                # Convert the response content to a PIL Image
                image = Image.open(io.BytesIO(response.content))
                
                # Save the image if a path is provided
                if save_path:
                    image.save(save_path)
                    print(f"Image saved to: {save_path}")
                
                return image
            else:
                return {"error": f"Request failed with status code: {response.status_code}"}
                
        except requests.RequestException as e:
            return {"error": str(e)}

def main():
    # Example usage
    client = OmniGenClient()
    
    # Test the connection
    print("Testing connection...")
    result = client.ping()
    print(f"Server response: {result}")
    
    # Generate an image
    print("\nGenerating image...")
    prompt = "a beautiful sunset over mountains"
    
    # You can now specify a custom save path
    save_path = "./output/my_generated_image.png"
    image = client.generate_image(prompt, save_path=save_path)
    
    if isinstance(image, Image.Image):
        print("Image generated successfully!")
        
        # You can also perform additional operations on the image
        # For example, display it (if running in a notebook):
        # image.show()
        
        # Or resize it:
        # resized_image = image.resize((512, 512))
        # resized_image.save("resized_image.png")
    else:
        print(f"Error generating image: {image.get('error')}")

if __name__ == "__main__":
    main()
