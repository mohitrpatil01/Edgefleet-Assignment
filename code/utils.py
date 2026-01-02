"""
Utility functions for cricket ball detection and tracking
EdgeFleet.AI Assessment - IIT BHU
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def visualize_annotations(csv_path, video_path=None, output_path=None, sample_frames=10):
    """
    Visualize annotations from CSV file.
    
    Args:
        csv_path: Path to annotation CSV
        video_path: Optional path to video for overlay
        output_path: Optional path to save visualization
        sample_frames: Number of frames to sample for visualization
    """
    df = pd.read_csv(csv_path)
    
    # Statistics
    total_frames = len(df)
    visible_frames = df[df['visible'] == 1].shape[0]
    detection_rate = 100 * visible_frames / total_frames
    
    print(f"Annotation Statistics:")
    print(f"  Total frames: {total_frames}")
    print(f"  Detected frames: {visible_frames}")
    print(f"  Detection rate: {detection_rate:.2f}%")
    print(f"  Missing frames: {total_frames - visible_frames}")
    
    # Plot trajectory
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Trajectory plot
    visible_data = df[df['visible'] == 1]
    axes[0, 0].plot(visible_data['x'], visible_data['y'], 'b-', alpha=0.5)
    axes[0, 0].scatter(visible_data['x'], visible_data['y'], c=visible_data['frame'], 
                       cmap='viridis', s=10)
    axes[0, 0].set_xlabel('X coordinate')
    axes[0, 0].set_ylabel('Y coordinate')
    axes[0, 0].set_title('Ball Trajectory')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(True, alpha=0.3)
    
    # X position over time
    axes[0, 1].plot(df['frame'], df['x'], 'r-', alpha=0.7)
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('X coordinate')
    axes[0, 1].set_title('Horizontal Position Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Y position over time
    axes[1, 0].plot(df['frame'], df['y'], 'g-', alpha=0.7)
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Y coordinate')
    axes[1, 0].set_title('Vertical Position Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Visibility over time
    axes[1, 1].fill_between(df['frame'], 0, df['visible'], alpha=0.5)
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('Visibility')
    axes[1, 1].set_title('Detection Visibility Over Time')
    axes[1, 1].set_ylim(-0.1, 1.1)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()

def calculate_trajectory_metrics(csv_path):
    """
    Calculate metrics about the ball trajectory.
    
    Args:
        csv_path: Path to annotation CSV
        
    Returns:
        Dictionary of metrics
    """
    df = pd.read_csv(csv_path)
    visible_data = df[df['visible'] == 1].copy()
    
    if len(visible_data) < 2:
        return {"error": "Insufficient visible frames"}
    
    # Calculate velocities
    visible_data['dx'] = visible_data['x'].diff()
    visible_data['dy'] = visible_data['y'].diff()
    visible_data['speed'] = np.sqrt(visible_data['dx']**2 + visible_data['dy']**2)
    
    # Calculate accelerations
    visible_data['ax'] = visible_data['dx'].diff()
    visible_data['ay'] = visible_data['dy'].diff()
    
    metrics = {
        'total_frames': len(df),
        'detected_frames': len(visible_data),
        'detection_rate': 100 * len(visible_data) / len(df),
        'avg_speed': visible_data['speed'].mean(),
        'max_speed': visible_data['speed'].max(),
        'trajectory_length': visible_data['speed'].sum(),
        'x_range': (visible_data['x'].min(), visible_data['x'].max()),
        'y_range': (visible_data['y'].min(), visible_data['y'].max()),
        'avg_x': visible_data['x'].mean(),
        'avg_y': visible_data['y'].mean(),
    }
    
    return metrics

def validate_csv_format(csv_path):
    """
    Validate that CSV file matches expected format.
    
    Args:
        csv_path: Path to annotation CSV
        
    Returns:
        Tuple (is_valid, error_message)
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Check columns
        required_cols = ['frame', 'x', 'y', 'visible']
        if list(df.columns) != required_cols:
            return False, f"Invalid columns. Expected {required_cols}, got {list(df.columns)}"
        
        # Check data types
        if not pd.api.types.is_integer_dtype(df['frame']):
            return False, "Frame column must be integer"
        
        if not pd.api.types.is_numeric_dtype(df['x']):
            return False, "X column must be numeric"
        
        if not pd.api.types.is_numeric_dtype(df['y']):
            return False, "Y column must be numeric"
        
        if not pd.api.types.is_integer_dtype(df['visible']):
            return False, "Visible column must be integer (0 or 1)"
        
        # Check visible values
        if not df['visible'].isin([0, 1]).all():
            return False, "Visible column must contain only 0 or 1"
        
        # Check invisible frames have -1 coordinates
        invisible = df[df['visible'] == 0]
        if not ((invisible['x'] == -1) & (invisible['y'] == -1)).all():
            return False, "Invisible frames must have x=-1 and y=-1"
        
        # Check frame indices
        if not (df['frame'] == range(len(df))).all():
            return False, "Frame indices must be sequential starting from 0"
        
        return True, "CSV format is valid"
        
    except Exception as e:
        return False, f"Error reading CSV: {str(e)}"

def extract_frames(video_path, output_dir, frame_indices=None, max_frames=None):
    """
    Extract specific frames from video for annotation or visualization.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        frame_indices: List of specific frame indices to extract (optional)
        max_frames: Maximum number of frames to extract (optional)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_indices is None:
        if max_frames:
            # Sample evenly distributed frames
            frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
        else:
            frame_indices = range(total_frames)
    
    extracted = 0
    frame_idx = 0
    
    print(f"Extracting frames from {video_path}")
    print(f"Total frames: {total_frames}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx in frame_indices:
            output_path = output_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(output_path), frame)
            extracted += 1
            
            if extracted % 10 == 0:
                print(f"Extracted {extracted} frames...")
        
        frame_idx += 1
    
    cap.release()
    print(f"Extraction complete. Saved {extracted} frames to {output_dir}")

def compare_annotations(csv_path1, csv_path2, output_path=None):
    """
    Compare two annotation files (e.g., ground truth vs predictions).
    
    Args:
        csv_path1: Path to first annotation CSV
        csv_path2: Path to second annotation CSV
        output_path: Optional path to save comparison plot
        
    Returns:
        Dictionary of comparison metrics
    """
    df1 = pd.read_csv(csv_path1)
    df2 = pd.read_csv(csv_path2)
    
    # Merge on frame
    merged = df1.merge(df2, on='frame', suffixes=('_1', '_2'))
    
    # Only compare frames where both have detections
    both_visible = merged[(merged['visible_1'] == 1) & (merged['visible_2'] == 1)]
    
    if len(both_visible) == 0:
        return {"error": "No overlapping detections"}
    
    # Calculate distances
    both_visible['distance'] = np.sqrt(
        (both_visible['x_1'] - both_visible['x_2'])**2 + 
        (both_visible['y_1'] - both_visible['y_2'])**2
    )
    
    metrics = {
        'total_frames': len(merged),
        'both_detected': len(both_visible),
        'only_1_detected': len(merged[merged['visible_1'] == 1]) - len(both_visible),
        'only_2_detected': len(merged[merged['visible_2'] == 1]) - len(both_visible),
        'avg_distance': both_visible['distance'].mean(),
        'max_distance': both_visible['distance'].max(),
        'median_distance': both_visible['distance'].median(),
    }
    
    # Plot comparison
    if output_path:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Trajectory comparison
        vis1 = merged[merged['visible_1'] == 1]
        vis2 = merged[merged['visible_2'] == 1]
        
        axes[0].plot(vis1['x_1'], vis1['y_1'], 'b-', label='Annotation 1', alpha=0.7)
        axes[0].plot(vis2['x_2'], vis2['y_2'], 'r-', label='Annotation 2', alpha=0.7)
        axes[0].set_xlabel('X coordinate')
        axes[0].set_ylabel('Y coordinate')
        axes[0].set_title('Trajectory Comparison')
        axes[0].legend()
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3)
        
        # Distance plot
        axes[1].plot(both_visible['frame'], both_visible['distance'], 'g-')
        axes[1].set_xlabel('Frame')
        axes[1].set_ylabel('Distance (pixels)')
        axes[1].set_title('Position Difference Over Time')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_path}")
        plt.close()
    
    return metrics

def create_gif_from_video(video_path, output_path, start_frame=0, end_frame=None, 
                          fps=10, scale=0.5):
    """
    Create a GIF from a video segment for easy sharing.
    
    Args:
        video_path: Path to video file
        output_path: Path to save GIF
        start_frame: Starting frame index
        end_frame: Ending frame index (None for video end)
        fps: Frames per second in output GIF
        scale: Scale factor for output size
    """
    from PIL import Image
    import imageio
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx >= start_frame:
            if end_frame is not None and frame_idx > end_frame:
                break
            
            # Resize and convert to RGB
            if scale != 1.0:
                new_width = int(frame.shape[1] * scale)
                new_height = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        frame_idx += 1
    
    cap.release()
    
    # Save as GIF
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"GIF saved to: {output_path}")
    print(f"Total frames in GIF: {len(frames)}")

if __name__ == "__main__":
    # Example usage
    print("Cricket Ball Detection Utilities")
    print("Import this module to use helper functions:")
    print("  - visualize_annotations()")
    print("  - calculate_trajectory_metrics()")
    print("  - validate_csv_format()")
    print("  - extract_frames()")
    print("  - compare_annotations()")
    print("  - create_gif_from_video()")
