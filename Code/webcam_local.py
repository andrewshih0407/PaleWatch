import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from torchvision import transforms, models
from PIL import Image

# ---------- DEVICE ----------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# ---------- CONFIG ----------
CHECKPOINT_PATH = "palewatch_v4_final.pth"
CONFIDENCE_THRESHOLD = 0.60     # lower than 0.75 so it stops getting stuck in WARNING
MARGIN_THRESHOLD = 0.12         # if classes are too close, call it WARNING
CAMERA_INDEX = 0

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ---------- MODEL ----------
def build_model(num_classes=2):
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

def load_model(path):
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

# ---------- CLASSIFICATION ----------
def classify_state(prob_bleached, prob_healthy):
    max_prob = max(prob_bleached, prob_healthy)
    diff = abs(prob_bleached - prob_healthy)

    # If both classes are weak OR too close together, call it WARNING
    if max_prob < CONFIDENCE_THRESHOLD or diff < MARGIN_THRESHOLD:
        return "WARNING"

    return "BLEACHED" if prob_bleached > prob_healthy else "HEALTHY"

def score_from_probs(prob_bleached, prob_healthy, state):
    # More human-readable score
    if state == "HEALTHY":
        return int(max(75, prob_healthy * 100))
    elif state == "BLEACHED":
        return int(min(35, prob_healthy * 100))
    else:
        return int(min(max(prob_healthy * 100, 40), 74))

def majority_vote(states):
    vals, counts = np.unique(states, return_counts=True)
    return vals[np.argmax(counts)]

def predict_frame(frame_bgr, model, tf, idx_to_class):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    x = tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    prob_dict = {idx_to_class[i]: float(probs[i]) for i in range(len(probs))}
    prob_bleached = prob_dict.get("bleached", 0.0)
    prob_healthy = prob_dict.get("healthy", 0.0)

    state = classify_state(prob_bleached, prob_healthy)
    score = score_from_probs(prob_bleached, prob_healthy, state)

    return state, score, prob_bleached, prob_healthy

# ---------- MAIN ----------
def main():
    model, tf, idx_to_class = load_model(CHECKPOINT_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    history = deque(maxlen=5)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        h, w, _ = frame.shape

        # Tighter crop so background does not dominate the prediction
        x1 = int(w * 0.35)
        y1 = int(h * 0.20)
        x2 = int(w * 0.65)
        y2 = int(h * 0.80)
        roi = frame[y1:y2, x1:x2]

        state, score, pb, ph = predict_frame(roi, model, tf, idx_to_class)
        history.append(state)
        smooth_state = majority_vote(list(history))

        # Recompute display score from smoothed state
        display_score = score
        if smooth_state == "HEALTHY":
            display_score = max(score, 75)
        elif smooth_state == "BLEACHED":
            display_score = min(score, 35)
        else:
            display_score = min(max(score, 40), 74)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, f"State: {smooth_state}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Score: {display_score}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Bleached: {pb:.3f}", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Healthy : {ph:.3f}", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("PaleWatch", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
