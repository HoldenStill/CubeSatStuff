import cv2
import numpy as np
import sys


def align_images(img1, img2):
    """
    Finds features in two images and aligns img2 to img1.
    """
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15

    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    matcher = cv2.DescriptorMatcher_create(
        cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    matches = sorted(matches, key=lambda x: x.distance)

    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    height, width = img1.shape
    img2_aligned = cv2.warpPerspective(img2, h, (width, height))

    return img2_aligned


def calculate_difference_cleaned(img_769, img_780, diff_thresh=5):
    """
    Detects fire using the difference metric and applies a
    two-step morphological process for robust cleaning.
    """
    diff_image = cv2.subtract(img_769, img_780)
    _, initial_mask = cv2.threshold(
        diff_image, diff_thresh, 255, cv2.THRESH_BINARY)

    # --- THE FIX: Two-Step Cleanup ---

    # 1. Remove small "salt" noise using an OPENING operation with a small kernel.
    small_kernel = np.ones((3, 3), np.uint8)
    noise_removed_mask = cv2.morphologyEx(
        initial_mask, cv2.MORPH_OPEN, small_kernel)

    # 2. Fill holes and consolidate the fire detection using a CLOSING operation with a larger kernel.
    large_kernel = np.ones((7, 7), np.uint8)
    final_mask = cv2.morphologyEx(
        noise_removed_mask, cv2.MORPH_CLOSE, large_kernel)

    # --- END FIX ---

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
        sys.exit()

    print("Aligning images...")
    img_780_aligned = align_images(img_769, img_780)
    print("Alignment complete.")

    final_mask = calculate_difference_cleaned(
        img_769, img_780_aligned, diff_thresh=200)

    # --- Display logic ---
    display_size = (500, 500)
    img_769_display = cv2.resize(img_769, display_size)
    img_780_display = cv2.resize(img_780_aligned, display_size)
    mask_display = cv2.resize(final_mask, display_size)

    img_769_display = cv2.cvtColor(img_769_display, cv2.COLOR_GRAY2BGR)
    img_780_display = cv2.cvtColor(img_780_display, cv2.COLOR_GRAY2BGR)
    mask_display = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)

    cv2.putText(img_769_display, '769nm Input', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img_780_display, '780nm Aligned', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(mask_display, 'Final Output', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    combined_view = np.hstack((img_769_display, img_780_display, mask_display))

    cv2.imshow('Wildfire Detection (Aligned Difference)', combined_view)

    print("Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
