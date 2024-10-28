import os
import torch
import torch.optim as optim
from torchvision.utils import save_image
import clip
from gan_model import Generator, Discriminator
from data_loader import get_data_loader
device = "cuda" if torch.cuda.is_available() else "cpu"
latent_dim = 100
MAX_CLIP_LENGTH = 77 

image_folder_path = "/kaggle/input/meme-data/2000_data"
csv_file_path = "/kaggle/input/testdata/2000_testdata.csv"
epochs = 500
batch_size = 16
lr = 0.0002
betas = (0.5, 0.999)


generator = Generator(latent_dim=latent_dim).to(device)
discriminator = Discriminator().to(device)
clip_model, preprocess = clip.load("ViT-B/32", device=device)
train_loader = get_data_loader(image_folder_path, csv_file_path, batch_size)


gen_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

def train():
    generator.train()
    discriminator.train()
    for epoch in range(epochs):
        for batch_idx, (real_images, captions) in enumerate(train_loader):
            
            real_images = real_images.to(device)
            with torch.no_grad():
                truncated_captions = [caption[:MAX_CLIP_LENGTH] for caption in captions]
                text_tokens = clip.tokenize(truncated_captions).to(device)
                caption_features = clip_model.encode_text(text_tokens)

            disc_optimizer.zero_grad()
            noise = torch.randn(real_images.size(0), latent_dim, device=device)
            fake_images = generator(noise)

            real_scores = discriminator(real_images)
            fake_scores = discriminator(fake_images.detach())

            disc_loss = -torch.mean(real_scores) + torch.mean(fake_scores)
            disc_loss.backward()
            disc_optimizer.step()

            gen_optimizer.zero_grad()
            gen_scores = discriminator(fake_images)
            gen_loss = -torch.mean(gen_scores)
            gen_loss.backward()
            gen_optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                      f"D Loss: {disc_loss.item()}, G Loss: {gen_loss.item()}")
                save_image(fake_images.data[:25], f"generated_samples/epoch_{epoch}_batch_{batch_idx}.png", nrow=5, normalize=True)
    torch.save(generator.state_dict(), 'trained_generator.pt')
    print("Model saved as trained_generator.pt")

if __name__ == "__main__":
    os.makedirs("generated_samples", exist_ok=True)
    train()