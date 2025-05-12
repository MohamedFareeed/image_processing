import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Try to import easyocr, but provide a fallback if it's not installed
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("EasyOCR is not installed. Please install it using:")
    print("pip install easyocr")
    print("Continuing with limited functionality...")

class LicensePlateRecognizer:
    def __init__(self, languages=['en']):
        """
        Initialize the license plate recognizer.
        
        Args:
            languages (list): List of languages to use for OCR. Default is English.
        """
        # Initialize EasyOCR reader if available
        if EASYOCR_AVAILABLE:
            try:
                self.reader = easyocr.Reader(languages, gpu=False)
                print("EasyOCR initialized successfully.")
            except Exception as e:
                print(f"Error initializing EasyOCR: {e}")
                print("OCR functionality will be disabled.")
                self.reader = None
        else:
            self.reader = None
            print("OCR functionality disabled. Install EasyOCR for text recognition.")
        
        # Load Haar cascade for license plate detection
        try:
            haar_cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
            if not os.path.exists(haar_cascade_path):
                print(f"Warning: Haar cascade file not found at {haar_cascade_path}")
                print("Using default detection method only.")
                self.plate_cascade = None
            else:
                self.plate_cascade = cv2.CascadeClassifier(haar_cascade_path)
                print("Haar cascade loaded successfully.")
        except Exception as e:
            print(f"Error loading Haar cascade: {e}")
            self.plate_cascade = None

    def preprocess_image(self, image):
        """
        Preprocess the image to enhance license plate detection.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to remove noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply adaptive thresholding to handle varying lighting conditions
        thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        return gray, filtered, thresh

    def detect_license_plates(self, image):
        """
        Detect license plates in the image using multiple methods.
        
        Args:
            image: Input image
            
        Returns:
            List of license plate regions (x, y, width, height)
        """
        plates = []
        height, width = image.shape[:2]
        
        # Method 1: Using contours and edge detection
        gray, filtered, thresh = self.preprocess_image(image)
        
        # Find edges
        edged = cv2.Canny(filtered, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        # Loop over the contours
        for contour in contours:
            # Approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
            
            # If the contour has 4 points, it's likely a license plate
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter based on aspect ratio and minimum size
                aspect_ratio = w / float(h)
                if 1.5 <= aspect_ratio <= 6.5 and w > width * 0.05 and h > height * 0.01:
                    plates.append((x, y, w, h))
        
        # Method 2: Using Haar Cascade (if available)
        if self.plate_cascade is not None:
            plate_detections = self.plate_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in plate_detections:
                # Filter based on aspect ratio
                aspect_ratio = w / float(h)
                if 1.5 <= aspect_ratio <= 6.5:
                    plates.append((x, y, w, h))
        
        # Remove duplicates by checking for overlaps
        filtered_plates = []
        for plate in plates:
            if not any(self._is_overlap(plate, existing) for existing in filtered_plates):
                filtered_plates.append(plate)
        
        return filtered_plates

    def _is_overlap(self, box1, box2, threshold=0.5):
        """
        Check if two bounding boxes overlap significantly.
        
        Args:
            box1: First bounding box (x, y, w, h)
            box2: Second bounding box (x, y, w, h)
            threshold: Overlap threshold
            
        Returns:
            True if boxes overlap significantly, False otherwise
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate area of intersection
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        intersection = x_overlap * y_overlap
        
        # Calculate area of both boxes
        area1 = w1 * h1
        area2 = w2 * h2
        
        # Calculate IoU (Intersection over Union)
        union = area1 + area2 - intersection
        iou = intersection / union if union > 0 else 0
        
        return iou > threshold

    def extract_text(self, image, plate_region):
        """
        Extract text from the license plate region.
        
        Args:
            image: Input image
            plate_region: Region of the license plate (x, y, w, h)
            
        Returns:
            Extracted text
        """
        # Check if OCR is available
        if self.reader is None:
            return "OCR_DISABLED"
        
        x, y, w, h = plate_region
        plate_img = image[y:y+h, x:x+w]
        
        # Enhance plate image for better OCR using basic techniques
        gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Simple enhancement methods (no CLAHE)
        # Apply basic thresholding
        _, thresh_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Try to improve contrast
        enhanced_plate = cv2.equalizeHist(gray_plate)
        
        # Use both enhanced versions with EasyOCR
        try:
            # Try with threshold image
            results1 = self.reader.readtext(thresh_plate)
            # Try with histogram equalized image
            results2 = self.reader.readtext(enhanced_plate)
            # Try with original grayscale image
            results3 = self.reader.readtext(gray_plate)
            
            # Combine results
            results = results1 + results2 + results3
            
            # Extract and clean the text
            texts = []
            for (_, text, confidence) in results:
                # Filter out low confidence detections
                if confidence > 0.3:
                    # Clean text (remove spaces, special characters)
                    cleaned_text = ''.join(c for c in text if c.isalnum())
                    if cleaned_text:
                        texts.append((cleaned_text, confidence))
            
            # Return the text with highest confidence
            if texts:
                texts.sort(key=lambda x: x[1], reverse=True)
                return texts[0][0]
            
        except Exception as e:
            print(f"Error during OCR: {e}")
            return "OCR_ERROR"
        
        return ""

    def process_image(self, image_path):
        """
        Process an image to detect license plates and extract text.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Original image with license plates marked and extracted texts
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            return None, []
        
        # Make a copy for drawing
        result_img = image.copy()
        
        # Detect license plates
        plate_regions = self.detect_license_plates(image)
        
        # Extract text from each plate
        results = []
        for i, plate_region in enumerate(plate_regions):
            x, y, w, h = plate_region
            
            # Extract text
            text = self.extract_text(image, plate_region)
            
            if text:
                # Draw rectangle around the plate
                cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Put text above the rectangle
                cv2.putText(result_img, text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                results.append({
                    'plate_text': text,
                    'position': plate_region
                })
        
        return result_img, results

    def process_video(self, video_path, output_path=None, display=True, sampling_rate=5):
        """
        Process a video to detect license plates and extract text.
        
        Args:
            video_path: Path to the input video
            output_path: Path to save the output video (optional)
            display: Whether to display the video during processing
            sampling_rate: Process every Nth frame to improve performance
            
        Returns:
            List of detected license plates with timestamps
        """
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video at {video_path}")
            return []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video writer if needed
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process video
        results = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every Nth frame to improve performance
            if frame_count % sampling_rate != 0:
                if out:
                    out.write(frame)
                continue
            
            # Get timestamp
            timestamp = frame_count / fps
            
            # Detect license plates
            plate_regions = self.detect_license_plates(frame)
            
            # Draw results on frame
            for i, plate_region in enumerate(plate_regions):
                x, y, w, h = plate_region
                
                # Extract text
                text = self.extract_text(frame, plate_region)
                
                if text:
                    # Draw rectangle around the plate
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Put text above the rectangle
                    cv2.putText(frame, text, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # Add to results
                    results.append({
                        'timestamp': timestamp,
                        'plate_text': text,
                        'position': plate_region
                    })
            
            # Write to output video
            if out:
                out.write(frame)
            
            # Display frame
            if display:
                cv2.imshow('License Plate Recognition', frame)
                
                # Exit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Release resources
        cap.release()
        if out:
            out.release()
        if display:
            cv2.destroyAllWindows()
        
        return results

# Example usage
if __name__ == "__main__":
    print("\n" + "="*50)
    print("LICENSE PLATE RECOGNITION SYSTEM")
    print("="*50)
    
    # Installation instructions
    print("\nPREREQUISITES:")
    print("1. Make sure you have installed all required packages:")
    print("   pip install opencv-python numpy matplotlib")
    print("2. For OCR functionality, install EasyOCR:")
    print("   pip install easyocr")
    print("\nNOTE: EasyOCR may take some time to install as it downloads models\n")
    
    # Initialize the recognizer
    print("Initializing License Plate Recognizer...")
    lpr = LicensePlateRecognizer(languages=['en'])
    
    # Interactive menu
    while True:
        print("\n" + "-"*50)
        print("MENU OPTIONS:")
        print("1. Process an image")
        print("2. Process a video")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            # Example 1: Process an image
            image_path = input("Enter the path to your image file: ")
            try:
                print(f"Processing image: {image_path}")
                result_img, detections = lpr.process_image(image_path)
                
                if result_img is not None:
                    # Display results
                    plt.figure(figsize=(10, 8))
                    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                    plt.title("License Plate Detection")
                    plt.axis('off')
                    plt.show()
                    
                    # Print detected plates
                    print(f"Detected {len(detections)} license plates:")
                    for i, detection in enumerate(detections):
                        print(f"  Plate {i+1}: {detection['plate_text']}")
                        
                    # Option to save the result
                    save_option = input("Do you want to save the result image? (y/n): ")
                    if save_option.lower() == 'y':
                        output_path = input("Enter output image path: ")
                        cv2.imwrite(output_path, result_img)
                        print(f"Result saved to: {output_path}")
            except Exception as e:
                print(f"Error processing image: {str(e)}")
        
        elif choice == '2':
            # Example 2: Process a video
            video_path = input("Enter the path to your video file: ")
            output_path = input("Enter the path for the output video (leave blank to skip saving): ")
            output_path = output_path if output_path.strip() else None
            
            try:
                print(f"Processing video: {video_path}")
                print("Press 'q' to stop processing")
                
                # Set display=True to see the processing in real-time
                sampling_rate = int(input("Enter sampling rate (1=every frame, 5=every 5th frame, etc.): ") or "5")
                detections = lpr.process_video(video_path, output_path, display=True, sampling_rate=sampling_rate)
                
                # Print detected plates
                print(f"\nDetected {len(detections)} license plates in video:")
                for i, detection in enumerate(detections):
                    time_str = f"{int(detection['timestamp'] // 60)}:{int(detection['timestamp'] % 60):02d}"
                    print(f"  Plate {i+1}: {detection['plate_text']} at {time_str}")
            except Exception as e:
                print(f"Error processing video: {str(e)}")
        
        elif choice == '3':
            print("Exiting program. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please select 1, 2, or 3.")