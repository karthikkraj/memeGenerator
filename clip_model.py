import clip
import torch
import os
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()

device = "mps" if torch.backends.mps.is_available() else "cpu"

clip_model, preprocess = clip.load("ViT-B/32", device=device)
def get_clip_score(image, text):
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_tokens = clip.tokenize([text]).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_tokens)
    
    similarity = torch.cosine_similarity(image_features, text_features)
    return similarity.item()