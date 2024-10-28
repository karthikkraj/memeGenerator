from PIL import ImageDraw, ImageFont
import torch
from gan_model import Generator
import clip
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
latent_dim = 100

generator = Generator(latent_dim).to(device)
generator.load_state_dict(torch.load("path_to_trained_generator.pth"))

def generate_meme(caption):
    z = torch.randn(1, latent_dim).to(device)
    with torch.no_grad():
        gen_img = generator(z)
    gen_img_pil = transforms.ToPILImage()(gen_img.squeeze().cpu()).convert("RGB")
    
    draw = ImageDraw.Draw(gen_img_pil)
    font = ImageFont.load_default()
    text_w, text_h = draw.textsize(caption, font)
    draw.text(((256 - text_w) / 2, 256 - text_h - 10), caption, font=font, fill=(255, 255, 255))
    
    gen_img_pil.save("generated_meme.png")
    gen_img_pil.show()

generate_meme("DATE WITH OVERLY ATTACHED GIRLFRIEND")