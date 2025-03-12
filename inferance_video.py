from ultralytics import YOLO
import torch
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

class VideoOFAInterface:
    def __init__(self, model_path, input_sizes=[320, 480, 640]):
        # Ensure the path is properly formatted
        model_path = os.path.normpath(model_path)
        print(f"Loading model from: {model_path}")
        
        self.model = YOLO(model_path)
        self.input_sizes = sorted(input_sizes)
        self.performance_metrics = {}
        
    def benchmark_video(self, video_path, max_frames=100, sample_interval=1):
        """
        Benchmark model performance on video with different input sizes
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process
            sample_interval: Process every Nth frame
            
        Returns:
            DataFrame with performance metrics
        """
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return None
            
        print(f"Benchmarking video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video file")
            return None
            
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video stats: {width}x{height}, {fps} FPS, {frame_count} frames")
        
        # Limit frames to process
        frames_to_process = min(frame_count, max_frames)
        print(f"Will process {frames_to_process} frames with interval {sample_interval}")
        
        # Store comprehensive metrics for each input size
        self.detailed_metrics = {
            'input_size': [],
            'inference_time': [],
            'fps': [],
            'avg_detections': [],
            'avg_confidence': [],
            'memory_usage_mb': []
        }
        
        # Extract frames
        print("Extracting frames...")
        frames = []
        frame_idx = 0
        pbar = tqdm(total=frames_to_process)
        
        while frame_idx < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % sample_interval == 0:
                frames.append(frame)
                pbar.update(1)
                
            frame_idx += 1
            
        pbar.close()
        cap.release()
        
        actual_frames = len(frames)
        print(f"Extracted {actual_frames} frames")
        
        # Process each input size
        for size in self.input_sizes:
            print(f"Benchmarking size {size}...")
            
            # Metrics for this size
            inference_times = []
            detection_counts = []
            confidences = []
            memory_usages = []
            
            # Process frames
            for frame in tqdm(frames, desc=f"Size {size}"):
                try:
                    # Track memory before inference
                    torch.cuda.empty_cache()  # Clear CUDA cache if using GPU
                    mem_before = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                    
                    # Run inference
                    start_time = time.time()
                    results = self.model(frame, imgsz=size)
                    inference_time = time.time() - start_time
                    
                    # Track memory after inference
                    mem_after = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                    memory_usage = mem_after - mem_before
                    
                    # Record metrics
                    inference_times.append(inference_time)
                    
                    # Count detections and confidences
                    if results and len(results) > 0:
                        boxes = results[0].boxes
                        detection_counts.append(len(boxes))
                        
                        if len(boxes) > 0:
                            frame_confidences = [float(box.conf[0]) for box in boxes]
                            confidences.extend(frame_confidences)
                    else:
                        detection_counts.append(0)
                        
                    memory_usages.append(memory_usage)
                    
                except Exception as e:
                    print(f"Error processing frame with size {size}: {e}")
            
            # Calculate average metrics
            avg_time = np.mean(inference_times) if inference_times else 0
            avg_detections = np.mean(detection_counts) if detection_counts else 0
            avg_confidence = np.mean(confidences) if confidences else 0
            avg_memory = np.mean(memory_usages) if memory_usages else 0
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            # Store in performance metrics
            self.performance_metrics[size] = {
                'inference_time': avg_time,
                'fps': fps,
                'avg_detections': avg_detections,
                'avg_confidence': avg_confidence,
                'memory_usage_mb': avg_memory
            }
            
            # Store detailed metrics
            self.detailed_metrics['input_size'].append(size)
            self.detailed_metrics['inference_time'].append(round(avg_time, 4))
            self.detailed_metrics['fps'].append(round(fps, 2))
            self.detailed_metrics['avg_detections'].append(round(avg_detections, 2))
            self.detailed_metrics['avg_confidence'].append(round(avg_confidence, 2))
            self.detailed_metrics['memory_usage_mb'].append(round(avg_memory, 2))
            
            print(f"  Size {size}: {avg_time:.4f} seconds ({fps:.2f} FPS), {avg_detections:.1f} detections/frame")
        
        # Create DataFrame
        df = pd.DataFrame(self.detailed_metrics)
        
        # Create sorted version for display
        display_df = df.sort_values('input_size').reset_index(drop=True)
        
        # Display table
        print("\n=== Video Performance Matrix ===")
        headers = [
            "Input Size", 
            "Inference Time (s)", 
            "FPS", 
            "Avg Detections/Frame", 
            "Avg Confidence",
            "Memory (MB)"
        ]
        
        try:
            table = tabulate(display_df, headers=headers, tablefmt="grid", showindex=False)
            print(table)
        except ImportError:
            print(display_df)
            
        return df
    
    def plot_performance_metrics(self, title="Video Performance Metrics"):
        """Generate comprehensive performance comparison charts"""
        df = pd.DataFrame(self.detailed_metrics)
        
        if df.empty:
            print("No performance data available to plot")
            return
        
        # Sort by input size for proper plotting
        df = df.sort_values('input_size').reset_index(drop=True)
        
        # Create a figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16)
        
        # Plot 1: FPS vs Input Size
        ax1 = axes[0, 0]
        ax1.plot(df['input_size'], df['fps'], 'o-', linewidth=2, markersize=8, color='blue')
        ax1.set_xlabel('Input Size')
        ax1.set_ylabel('FPS')
        ax1.set_title('Speed vs Input Size')
        ax1.grid(True)
        
        # Add value labels
        for i, txt in enumerate(df['fps']):
            ax1.annotate(f"{txt}", 
                        (df['input_size'].iloc[i], df['fps'].iloc[i]),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
        
        # Plot 2: Confidence vs Input Size
        ax2 = axes[0, 1]
        ax2.plot(df['input_size'], df['avg_confidence'], 'o-', color='green', linewidth=2, markersize=8)
        ax2.set_xlabel('Input Size')
        ax2.set_ylabel('Average Confidence')
        ax2.set_title('Accuracy vs Input Size')
        ax2.grid(True)
        
        # Add value labels
        for i, txt in enumerate(df['avg_confidence']):
            ax2.annotate(f"{txt}", 
                        (df['input_size'].iloc[i], df['avg_confidence'].iloc[i]),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
        
        # Plot 3: Detections vs Input Size
        ax3 = axes[1, 0]
        ax3.plot(df['input_size'], df['avg_detections'], 'o-', color='purple', linewidth=2, markersize=8)
        ax3.set_xlabel('Input Size')
        ax3.set_ylabel('Average Detections/Frame')
        ax3.set_title('Detections vs Input Size')
        ax3.grid(True)
        
        # Add value labels
        for i, txt in enumerate(df['avg_detections']):
            ax3.annotate(f"{txt}", 
                        (df['input_size'].iloc[i], df['avg_detections'].iloc[i]),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
                        
        # Plot 4: Memory Usage vs Input Size
        ax4 = axes[1, 1]
        ax4.plot(df['input_size'], df['memory_usage_mb'], 'o-', color='orange', linewidth=2, markersize=8)
        ax4.set_xlabel('Input Size')
        ax4.set_ylabel('Memory Usage (MB)')
        ax4.set_title('Memory vs Input Size')
        ax4.grid(True)
        
        # Add value labels
        for i, txt in enumerate(df['memory_usage_mb']):
            ax4.annotate(f"{txt}", 
                        (df['input_size'].iloc[i], df['memory_usage_mb'].iloc[i]),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the plot
        plt.savefig("video_performance_comparison.png", dpi=300)
        
        plt.show()
        print("Performance comparison chart saved as video_performance_comparison.png")
    
    def select_optimal_size(self, speed_requirement=None):
        """Select the optimal input size based on speed requirements"""
        if not self.performance_metrics:
            print("No performance metrics available. Run benchmark_video first.")
            return self.input_sizes[-1]
            
        if speed_requirement is None:
            # Return largest size if no speed requirement
            return self.input_sizes[-1]
            
        # Find the largest size that meets speed requirement
        for size in sorted(self.input_sizes, reverse=True):
            if self.performance_metrics[size]['inference_time'] <= speed_requirement:
                return size
                
        # If no size meets requirement, return smallest
        return self.input_sizes[0]
    
    def process_video(self, input_video, output_video=None, size=None, speed_requirement=None, show_fps=True):
        """
        Process a video with the selected input size
        
        Args:
            input_video: Path to input video
            output_video: Path to output video (if None, will be auto-generated)
            size: Input size to use (if None, will use the largest size or select based on speed_requirement)
            speed_requirement: Maximum inference time per frame (seconds)
            show_fps: Whether to display FPS counter on the output video
            
        Returns:
            Path to the output video
        """
        if not os.path.exists(input_video):
            print(f"Input video not found: {input_video}")
            return None
            
        # Select input size
        if size is None:
            size = self.select_optimal_size(speed_requirement)
        
        print(f"Processing video with input size {size}...")
        
        # Generate output video path if not provided
        if output_video is None:
            base_name, ext = os.path.splitext(input_video)
            output_video = f"{base_name}_processed_{size}{ext}"
            
        # Open input video
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print("Error opening input video")
            return None
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {frame_count} frames")
        
        # Initialize output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("Error opening output video writer")
            cap.release()
            return None
            
        # Process frames
        frame_idx = 0
        processing_times = []
        
        pbar = tqdm(total=frame_count, desc="Processing video")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            start_time = time.time()
            results = self.model(frame, imgsz=size)
            process_time = time.time() - start_time
            processing_times.append(process_time)
            
            # Draw results on frame
            result_frame = results[0].plot()
            
            # Calculate FPS (based on moving average of last 30 frames)
            recent_times = processing_times[-30:]
            avg_time = sum(recent_times) / len(recent_times)
            current_fps = 1.0 / avg_time if avg_time > 0 else 0
            
            # Add FPS counter if requested
            if show_fps:
                cv2.putText(result_frame, f"Size: {size}, FPS: {current_fps:.1f}", 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add detection count
                boxes = results[0].boxes
                cv2.putText(result_frame, f"Detections: {len(boxes)}", 
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write frame to output video
            out.write(result_frame)
            
            frame_idx += 1
            pbar.update(1)
            
        # Clean up
        pbar.close()
        cap.release()
        out.release()
        
        # Calculate overall stats
        avg_process_time = np.mean(processing_times)
        avg_fps = 1.0 / avg_process_time if avg_process_time > 0 else 0
        
        print(f"Video processing complete.")
        print(f"Average processing time: {avg_process_time:.4f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Output saved to: {output_video}")
        
        return output_video
    
    def export_metrics_to_excel(self, filename="video_ofa_metrics.xlsx"):
        """Export performance metrics to Excel file"""
        if not self.detailed_metrics:
            print("No metrics to export. Run benchmark_video first.")
            return
            
        df = pd.DataFrame(self.detailed_metrics)
        df = df.sort_values('input_size').reset_index(drop=True)
        
        try:
            df.to_excel(filename, index=False)
            print(f"Metrics exported to {filename}")
        except Exception as e:
            print(f"Error exporting to Excel: {e}")
            # Fallback to CSV
            csv_filename = filename.replace('.xlsx', '.csv')
            df.to_csv(csv_filename, index=False)
            print(f"Metrics exported to {csv_filename}")


# Main execution
if __name__ == "__main__":
    # Use raw string for path to avoid escape character issues
    model_path = r"D:\personal\study_Sem1\Thesis\Learning\OFA\OFA_YOLO\runs\yolov8_ofa_project\small_test\weights\best.pt"
    
    # Check if the file exists
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        # Try to find model files
        base_dir = r"D:\personal\study_Sem1\Thesis\Learning\OFA\OFA_YOLO\runs"
        if os.path.exists(base_dir):
            print("Searching for model files...")
            found_models = []
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.endswith('.pt'):
                        found_models.append(os.path.join(root, file))
            
            if found_models:
                print("Found these model files:")
                for i, model in enumerate(found_models):
                    print(f"{i+1}. {model}")
                
                # Use the first model found
                model_path = found_models[0]
                print(f"Using model: {model_path}")
            else:
                print("No model files found.")
                exit(1)
    
    # Define input sizes to test
    input_sizes = [160, 320, 480, 640]
    
    # Initialize interface
    interface = VideoOFAInterface(model_path, input_sizes=input_sizes)
    
    # Path to your video file
    video_path = r"D:\personal\study_Sem1\Thesis\Learning\OFA\OFA_YOLO\shapes.mp4"
    
    # Ask user for video path if not found
    if not os.path.exists(video_path):
        video_path = input("Enter path to video file: ")
        
    # Benchmark different sizes
    metrics_df = interface.benchmark_video(video_path, max_frames=100, sample_interval=2)
    
    # Plot performance metrics
    interface.plot_performance_metrics()
    
    # Export metrics to Excel
    interface.export_metrics_to_excel()
    
    # Process video with automatic size selection
    print("\nProcessing video with different sizes:")
    
    # Process with highest quality (largest size)
    interface.process_video(video_path, output_video="output_high_quality.mp4", 
                          size=max(input_sizes), show_fps=True)
    
    # Process with balanced setting
    mid_size = input_sizes[len(input_sizes)//2]
    interface.process_video(video_path, output_video="output_balanced.mp4", 
                          size=mid_size, show_fps=True)
    
    # Process with speed priority (30 FPS requirement)
    fps_requirement = 1.0/30  # 30 FPS = 0.033s per frame
    optimal_size = interface.select_optimal_size(speed_requirement=fps_requirement)
    interface.process_video(video_path, output_video="output_speed_optimized.mp4", 
                          size=optimal_size, show_fps=True)
    
    print("\nSummary of processed videos:")
    print(f"1. High Quality: Size {max(input_sizes)}px")
    print(f"2. Balanced: Size {mid_size}px")
    print(f"3. Speed Optimized: Size {optimal_size}px (for 30 FPS target)")