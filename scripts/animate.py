import torch
from PIL import Image
from animate_diff.animate import AnimateDiffPipeline
import imageio
import cv2

def generate_motion(image_path, video_path, output_path, prompt):
    # Load image
    input_image = Image.open(image_path).convert("RGB")

    # Load AnimateDiff model
    pipe = AnimateDiffPipeline.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    # Extract frames from reference video
    cap = cv2.VideoCapture(video_path)
    reference_frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        reference_frames.append(Image.fromarray(frame))
    cap.release()

    # Limit frames for safety
    reference_frames = reference_frames[:16]

    # Generate animation
    animated_frames = pipe(
        prompt=prompt,
        reference_frames=reference_frames,
        num_frames=len(reference_frames),
        guidance_scale=7.5
    ).frames

    # Save output
    imageio.mimsave(output_path, animated_frames, fps=12)
    print(f"✅ Saved: {output_path}")
