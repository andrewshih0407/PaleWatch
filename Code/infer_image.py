import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def build_model(num_classes=2):
    # weights=None so it does NOT try to download ImageNet weights locally
    m = models.efficientnet_b0(weights=None)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.BatchNorm1d(in_f),
        nn.Dropout(p=0.5),
        nn.Linear(in_f, 512),
        nn.SiLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return m

def load_checkpoint(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    class_to_idx = ckpt["class_to_idx"]
    img_size = ckpt["img_size"]

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = build_model(num_classes=len(class_to_idx))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    return model, tf, idx_to_class

def classify_state(prob_bleached, prob_healthy):
    max_prob = max(prob_bleached, prob_healthy)
    if max_prob < 0.75:
        return "warning"
    return "bleached" if prob_bleached > prob_healthy else "healthy"

def predict_image(model, tf, idx_to_class, img_path):
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    prob_dict = {idx_to_class[i]: float(probs[i]) for i in range(len(probs))}
    prob_bleached = prob_dict.get("bleached", 0.0)
    prob_healthy = prob_dict.get("healthy", 0.0)

    state = classify_state(prob_bleached, prob_healthy)

    print(f"Device   : {DEVICE}")
    print(f"Bleached : {prob_bleached:.3f}")
    print(f"Healthy  : {prob_healthy:.3f}")
    print(f"State    : {state}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python infer_image.py <checkpoint.pth> <image.jpg>")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    img_path = sys.argv[2]

    model, tf, idx_to_class = load_checkpoint(ckpt_path)
    predict_image(model, tf, idx_to_class, img_path)
