from ultralytics import YOLO
import torch
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from tabulate import tabulate

class OFAModelInterface:
    def __init__(self, model_path, input_sizes=[320, 480, 640]):
        # Ensure the path is properly formatted
        model_path = os.path.normpath(model_path)
        print(f"Loading model from: {model_path}")
        
        self.model = YOLO(model_path)
        self.input_sizes = sorted(input_sizes)
        self.performance_metrics = {}
        
        # Benchmark each size
        self._benchmark_sizes()
        
    def _benchmark_sizes(self):
        # Sample image for benchmarking - use raw strings to avoid escape issues
        sample_image1 = r"D:\personal\study_Sem1\Thesis\Learning\OFA\OFA_YOLO\shapes\test\images\shape_14.png"
        sample_image2 = r"D:\personal\study_Sem1\Thesis\Learning\OFA\OFA_YOLO\shapes\test\images\shape_98.png"
        
        print(f"Benchmarking with sample images")
        
        # Store comprehensive metrics
        self.detailed_metrics = {
            'input_size': [],
            'inference_time': [],
            'fps': [],
            'num_detections': [],
            'avg_confidence': []
        }
        
        for size in self.input_sizes:
            print(f"Benchmarking size {size}...")
            try:
                # Run warmup
                self.model(sample_image1, imgsz=size)
                
                # Benchmark with multiple runs for stability
                num_runs = 5
                times = []
                detection_counts = []
                confidences = []
                
                for _ in range(num_runs):
                    start_time = time.time()
                    results = self.model(sample_image2, imgsz=size)
                    end_time = time.time()
                    
                    inference_time = end_time - start_time
                    times.append(inference_time)
                    
                    # Count detections and collect confidences
                    if results and len(results) > 0:
                        boxes = results[0].boxes
                        detection_counts.append(len(boxes))
                        if len(boxes) > 0:
                            confs = [float(box.conf[0]) for box in boxes]
                            confidences.extend(confs)
                
                # Calculate average metrics
                avg_time = np.mean(times)
                avg_detections = np.mean(detection_counts) if detection_counts else 0
                avg_confidence = np.mean(confidences) if confidences else 0
                fps = 1.0 / avg_time if avg_time > 0 else 0
                
                # Store in performance metrics
                self.performance_metrics[size] = avg_time
                
                # Store detailed metrics
                self.detailed_metrics['input_size'].append(size)
                self.detailed_metrics['inference_time'].append(round(avg_time, 4))
                self.detailed_metrics['fps'].append(round(fps, 2))
                self.detailed_metrics['num_detections'].append(avg_detections)
                self.detailed_metrics['avg_confidence'].append(round(avg_confidence, 2))
                
                print(f"  Size {size}: {avg_time:.4f} seconds ({fps:.2f} FPS)")
            except Exception as e:
                print(f"Error benchmarking size {size}: {e}")
                self.performance_metrics[size] = float('inf')  # Use infinity if benchmarking fails
        
    def select_optimal_size(self, speed_requirement=None):
        if speed_requirement is None:
            # Return largest size if no speed requirement
            return self.input_sizes[-1]
            
        # Find the largest size that meets speed requirement
        for size in sorted(self.input_sizes, reverse=True):
            if self.performance_metrics[size] <= speed_requirement:
                return size
                
        # If no size meets requirement, return smallest
        return self.input_sizes[0]
    
    def display_performance_matrix(self):
        """Display a formatted performance matrix"""
        # Create DataFrame from detailed metrics
        df = pd.DataFrame(self.detailed_metrics)
        
        # Sort by input size
        df = df.sort_values('input_size')
        
        # Print nice table
        print("\n=== Performance Matrix ===")
        headers = [
            "Input Size", 
            "Inference Time (s)", 
            "FPS", 
            "Avg Detections", 
            "Avg Confidence"
        ]
        
        try:
            table = tabulate(df, headers=headers, tablefmt="grid", showindex=False)
            print(table)
        except ImportError:
            print(df)  # Fallback if tabulate is not installed
            
        # Return the dataframe for potential further use
        return df
        
    def predict(self, image_path, speed_requirement=None, visualize=True):
        """
        Run prediction with the optimal model size
        
        Args:
            image_path: Path to the image
            speed_requirement: Maximum acceptable inference time (seconds)
            visualize: Whether to display the detection results
        
        Returns:
            results: Detection results
            selected_size: The input size that was used
        """
        size = self.select_optimal_size(speed_requirement)
        print(f"Selected size {size} for inference")
        
        # Run inference
        results = self.model(image_path, imgsz=size)
        
        # Visualization
        if visualize:
            self.visualize_detection(image_path, results, size)
            
        return results, size
    
    def visualize_detection(self, image_path, results, size):
        """
        Visualize detection results
        
        Args:
            image_path: Path to the original image
            results: Detection results from YOLO
            size: Input size used for this detection
        """
        # Load original image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get the annotated image from results
        annotated_img = results[0].plot()
        
        # Create figure with two subplots
        plt.figure(figsize=(12, 6))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis("off")
        
        # Annotated image
        plt.subplot(1, 2, 2)
        plt.imshow(annotated_img)
        plt.title(f"Detection Results (Size: {size}×{size})")
        plt.axis("off")
        
        # Add detection info
        boxes = results[0].boxes
        info_text = f"Detections: {len(boxes)}"
        if len(boxes) > 0:
            avg_conf = sum(float(box.conf[0]) for box in boxes) / len(boxes)
            info_text += f"\nAvg Confidence: {avg_conf:.2f}"
        plt.figtext(0.5, 0.01, info_text, ha="center", fontsize=12, 
                    bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        plt.tight_layout()
        
        # Save visualization
        save_dir = os.path.dirname(image_path)
        save_path = os.path.join(save_dir, f"detection_result_{size}.png")
        plt.savefig(save_path)
        
        plt.show()
        print(f"Visualization saved to {save_path}")
    
    def plot_performance_comparison(self):
        """Generate a performance comparison chart"""
        df = pd.DataFrame(self.detailed_metrics)
        
        if df.empty:
            print("No performance data available to plot")
            return
        
        # Create a figure with two subplots (1 row, 2 columns)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: FPS vs Input Size
        ax1.plot(df['input_size'], df['fps'], 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Input Size')
        ax1.set_ylabel('FPS')
        ax1.set_title('Speed vs Input Size')
        ax1.grid(True)
        
        # Add value labels
        for i, txt in enumerate(df['fps']):
            ax1.annotate(f"{txt}", 
                        (df['input_size'][i], df['fps'][i]),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
        
        # Plot 2: Confidence vs Input Size
        ax2.plot(df['input_size'], df['avg_confidence'], 'o-', color='green', linewidth=2, markersize=8)
        ax2.set_xlabel('Input Size')
        ax2.set_ylabel('Average Confidence')
        ax2.set_title('Accuracy vs Input Size')
        ax2.grid(True)
        
        # Add value labels
        for i, txt in enumerate(df['avg_confidence']):
            ax2.annotate(f"{txt}", 
                        (df['input_size'][i], df['avg_confidence'][i]),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig("performance_comparison.png", dpi=300)
        
        plt.show()
        print("Performance comparison chart saved as performance_comparison.png")


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
    
    # Define multiple input sizes to test
    input_sizes = [160, 320, 480, 640]
    
    # Initialize interface
    interface = OFAModelInterface(model_path, input_sizes=input_sizes)
    
    # Display performance matrix
    perf_df = interface.display_performance_matrix()
    
    # Plot performance comparison
    interface.plot_performance_comparison()
    
    # Test image
    test_image = r"D:\personal\study_Sem1\Thesis\Learning\OFA\OFA_YOLO\shapes\test\images\shape_887.png"
    
    # Fast prediction with visualization
    results, selected_size = interface.predict(test_image, speed_requirement=0.05, visualize=True)
    print(f"Used size {selected_size}×{selected_size} for detection")
    
    # Print detection results
    for r in results:
        boxes = r.boxes
        print(f"Detected {len(boxes)} objects")
        
        # Print class and confidence for each detection
        if len(boxes) > 0:
            # Create a simple table for detections
            detection_data = []
            for i, box in enumerate(boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                coords = box.xyxy[0].tolist()  # Get coordinates [x1, y1, x2, y2]
                detection_data.append([i+1, cls, f"{conf:.2f}", f"{coords}"])
            
            # Display detection table
            try:
                detection_table = tabulate(detection_data, 
                                         headers=["#", "Class", "Confidence", "Coordinates"],
                                         tablefmt="grid")
                print("\n=== Detection Results ===")
                print(detection_table)
            except ImportError:
                # Fallback if tabulate isn't installed
                for i, data in enumerate(detection_data):
                    print(f"Detection {i+1}: Class={data[1]}, Conf={data[2]}, Coords={data[3]}")