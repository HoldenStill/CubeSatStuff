import cv2
import numpy as np
import sys


def calculate_k_index_robust(img_769, img_780, brightness_thresh=200, ratio_thresh=6.0):
    """
    Calculates a fire mask using a more robust two-factor method.
    This version includes a fix for the data type mismatch error.
    """
    _, brightness_mask = cv2.threshold(
        img_769, brightness_thresh, 255, cv2.THRESH_BINARY)

    img_769_float = img_769.astype(np.float32)
    img_780_float = img_780.astype(np.float32)
    epsilon = 1e-7
    ratio_image = np.divide(img_769_float, img_780_float + epsilon)

    # ratio_mask is created here as a float32 array because ratio_image is float32
    _, ratio_mask_float = cv2.threshold(
        ratio_image, ratio_thresh, 255, cv2.THRESH_BINARY)

    # --- THE FIX ---
    # Convert the float mask to an 8-bit integer (uint8) to match brightness_mask.
    ratio_mask = ratio_mask_float.astype(np.uint8)
    # --- END FIX ---

    # Now both masks are uint8, so the bitwise operation will succeed.
    final_mask = cv2.bitwise_and(brightness_mask, ratio_mask)

    return final_mask


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
        print(
            f"Please make sure the folder 'img-proc-4-wildfire' is in the same directory.")
        sys.exit()

    # Define a single size for all processing and display.
    processing_size = (500, 500)

    img_769_aligned = cv2.resize(img_769, processing_size)
    img_780_aligned = cv2.resize(img_780, processing_size)

    # Apply the robust K-Index logic on the ALIGNED images.
    k_index_mask = calculate_k_index_robust(img_769_aligned, img_780_aligned)

    # --- Display logic ---
    img_769_display = cv2.cvtColor(img_769_aligned, cv2.COLOR_GRAY2BGR)
    img_780_display = cv2.cvtColor(img_780_aligned, cv2.COLOR_GRAY2BGR)
    mask_display = cv2.cvtColor(k_index_mask, cv2.COLOR_GRAY2BGR)

    cv2.putText(img_769_display, '769nm Input', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img_780_display, '780nm Input', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(mask_display, 'K-index output', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    combined_view = np.hstack((img_769_display, img_780_display, mask_display))

    cv2.imshow('Wildfire Detection (Robust Method)', combined_view)

    print("Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
