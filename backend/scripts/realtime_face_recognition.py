import cv2
import requests
import numpy as np
import time
import sys

# API Configuration
API_URL = "http://127.0.0.1:8000/detect/faces?recognize_faces=true"

def main():
    # 1. Webcam Capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
        
    print("NOVA Real-Time Face Recognition started.")
    print("Press 'q' to exit.")

    try:
        while True:
            # 2. Frame Processing Loop
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # 3. Encode Frame
            success, encoded_image = cv2.imencode(".jpg", frame)
            if not success:
                continue

            # 4. Send Frame to API
            try:
                files = {"file": ("frame.jpg", encoded_image.tobytes(), "image/jpeg")}
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    faces = data.get("faces", [])
                    
                    # 5. Parse API Response and 6. Draw Detection Results
                    for face in faces:
                        bbox = face.get("bbox", {})
                        confidence = face.get("confidence", 0.0)
                        person_id = face.get("person_id")
                        
                        # Convert normalized bbox to pixel coordinates
                        h, w = frame.shape[:2]
                        left = int(bbox.get("left", 0) * w)
                        top = int(bbox.get("top", 0) * h)
                        right = int(bbox.get("right", 0) * w)
                        bottom = int(bbox.get("bottom", 0) * h)
                        
                        # Determine label
                        if person_id:
                            label = f"{person_id} ({confidence:.2f})"
                            color = (0, 255, 0) # Green for recognized
                        else:
                            label = f"Unknown ({confidence:.2f})"
                            color = (0, 0, 255) # Red for unknown
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        
                        # Draw label
                        cv2.putText(frame, label, (left, top - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    print(f"API Error: {response.status_code} - {response.text}")
            
            except requests.exceptions.RequestException as e:
                print(f"Connection Error: {e}")

            # 7. Display Video Stream
            cv2.imshow("NOVA Real-Time Face Recognition", frame)

            # 8. Exit Condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("NOVA Real-Time Face Recognition stopped.")

if __name__ == "__main__":
    main()
