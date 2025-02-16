import cv2
import os

def create_video_from_images(image_folder, output_video_path, fps=30):
    # Get all .png files in the folder, sorted by filename
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])

    if not images:
        raise ValueError("No .png images found in the specified folder!")

    # Read the first image to get its dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Warning: Skipping invalid image {image_path}")
            continue
        out.write(frame)  # Add the frame to the video

    # Release the VideoWriter
    out.release()
    print(f"Video saved at: {output_video_path}")

# Example usage
image_folder = './visualization/edge/video'  # Path to the folder containing .png images
output_video_path = './visualization/edge/output_video.mp4'  # Path to save the output video
fps = 12  # Frames per second

create_video_from_images(image_folder, output_video_path, fps)
