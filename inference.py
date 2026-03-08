import cv2
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from facenet_pytorch import MTCNN
from model_utils import get_model, get_transforms

class DeepfakeDetector:
    def __init__(self, model_path, device='cuda'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.face_detector = MTCNN(
            device=self.device
        )

        # Load the trained model
        self.model = get_model(weights_path=model_path).to(self.device)
        self.transform = get_transforms()

    def predict_image(self, image_path):
        img = Image.open(image_path).convert('RGB')
        
        # Detect and crop faces
        face = self.face_detector(img)

        if face is None:
            print(f"Skipping {image_path}: No face detected")
            return None

        # MTCNN returns a tensor; transform expects PIL
        if torch.is_tensor(face):
            face = to_pil_image(face)
        img_tensor = self.transform(face).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)
            probability = torch.softmax(output, dim=1)[0][1].item()
        return probability
    
    def predict_video(self, video_path, sample_rate=10):
        cap = cv2.VideoCapture(video_path)
        scores = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Only process every 10th frame to save computation
            if frame_count % sample_rate == 0:
                # Convert CV@BGR to PIL RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                face = self.face_detector(pil_img)

                if face is not None:
                    if torch.is_tensor(face):
                        face = to_pil_image(face)
                    img_tensor = self.transform(face).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        output = self.model(img_tensor)
                        scores.append(torch.softmax(output, dim=1)[0][1].item())
        
            frame_count += 1
        cap.release()

        return sum(scores) / len(scores) if scores else 0
