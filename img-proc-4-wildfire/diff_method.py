import cv2
import numpy as np
import sys

# --- NEW Difference Metric Function ---


def calculate_difference_metric(img_769, img_780, threshold_level=200):
    """
    Calculates a fire mask by subtracting the background (780nm) from the 
    signal (769nm) image, as described in the documents.
    """
    # The core logic is to subtract the background image from the signal image.
    # cv2.subtract handles pixel values that would go below 0 by clamping them at 0.
    diff_image = cv2.subtract(img_769, img_780)

    # Any pixel where the difference is greater than the threshold is considered fire.
    # This threshold may need to be tuned for real-world images.
    _, fire_mask = cv2.threshold(
        diff_image, threshold_level, 255, cv2.THRESH_BINARY)

    return fire_mask


# --- Main execution block ---
if __name__ == "__main__":
    image_769_path = "img-proc-4-wildfire/769nm.png"
    image_780_path = "img-proc-4-wildfire/780nm.png"

    try:
        img_769 = cv2.imread(image_769_path, cv2.IMREAD_GRAYSCALE)
        img_780 = cv2.imread(image_780_path, cv2.IMREAD_GRAYSCALE)
        if img_769 is None or img_780 is None:
            raise FileNotFoundError("One or both image files were not found.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please make sure the folder 'img-proc-4-wildfire' is in the same directory as the script.")
        sys.exit()

    # Apply the NEW Difference Metric logic.
    difference_mask = calculate_difference_metric(img_769, img_780)

    # --- Prepare images for display ---
    img_769_bgr = cv2.cvtColor(img_769, cv2.COLOR_GRAY2BGR)
    img_780_bgr = cv2.cvtColor(img_780, cv2.COLOR_GRAY2BGR)
    mask_bgr = cv2.cvtColor(difference_mask, cv2.COLOR_GRAY2BGR)

    display_size = (450, 450)
    img_769_resized = cv2.resize(img_769_bgr, display_size)
    img_780_resized = cv2.resize(img_780_bgr, display_size)
    mask_resized = cv2.resize(mask_bgr, display_size)

    cv2.putText(img_769_resized, '769nm Input', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(img_780_resized, '780nm Input', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(mask_resized, 'Difference Output', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    combined_view = np.hstack((img_769_resized, img_780_resized, mask_resized))

    # Updated the window title
    cv2.imshow('Wildfire Detection (Difference Method)', combined_view)

    print("Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
