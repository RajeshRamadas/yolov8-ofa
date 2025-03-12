from ultralytics import YOLO
import torch
import time
import os

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
        
        for size in self.input_sizes:
            print(f"Benchmarking size {size}...")
            try:
                # Run warmup
                self.model(sample_image1, imgsz=size)
                
                # Benchmark
                start_time = time.time()
                self.model(sample_image2, imgsz=size)
                inference_time = time.time() - start_time
                
                self.performance_metrics[size] = inference_time
                print(f"  Size {size}: {inference_time:.4f} seconds")
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
        
    def predict(self, image_path, speed_requirement=None):
        size = self.select_optimal_size(speed_requirement)
        print(f"Selected size {size} for inference")
        results = self.model(image_path, imgsz=size)
        return results, size

# Usage
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
    
    interface = OFAModelInterface(model_path, input_sizes=[320, 480, 640])
    
    # Test image
    test_image = r"D:\personal\study_Sem1\Thesis\Learning\OFA\OFA_YOLO\shapes\test\images\shape_887.png"
    
    # Fast prediction (adapt to hardware constraints)
    results, selected_size = interface.predict(test_image, speed_requirement=0.05)  # 50ms max
    print(f"Used size {selected_size}×{selected_size} for detection")
    
    # Print detection results
    for r in results:
        boxes = r.boxes
        print(f"Detected {len(boxes)} objects")
        
        # Print class and confidence for each detection
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()  # Get coordinates [x1, y1, x2, y2]
            print(f"Class: {cls}, Confidence: {conf:.2f}, Coords: {coords}")