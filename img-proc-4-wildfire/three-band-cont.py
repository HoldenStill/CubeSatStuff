import cv2
import numpy as np
import sys


def align_images(img_reference, img_to_align):
    """
    Finds features and aligns img_to_align to img_reference's perspective.
    """
    MAX_FEATURES = 1000
    GOOD_MATCH_PERCENT = 0.15

    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints_ref, descriptors_ref = orb.detectAndCompute(img_reference, None)
    keypoints_align, descriptors_align = orb.detectAndCompute(
        img_to_align, None)

    matcher = cv2.DescriptorMatcher_create(
        cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors_align, descriptors_ref, None)
    matches = sorted(matches, key=lambda x: x.distance)

    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    points_ref = np.zeros((len(matches), 2), dtype=np.float32)
    points_align = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points_ref[i, :] = keypoints_ref[match.trainIdx].pt
        points_align[i, :] = keypoints_align[match.queryIdx].pt

    h, mask = cv2.findHomography(points_align, points_ref, cv2.RANSAC)

    height, width = img_reference.shape
    img_aligned = cv2.warpPerspective(img_to_align, h, (width, height))

    return img_aligned


def calculate_continuum_removal(img_peak, img_left, img_right, spike_thresh=50):
    """
    Isolates the K-spike by subtracting the background continuum, then cleans the result.
    """
    # Convert images to float for accurate calculations.
    peak_f = img_peak.astype(np.float32)
    left_f = img_left.astype(np.float32)
    right_f = img_right.astype(np.float32)

    # 1. Calculate the continuum by averaging the two "shoulder" bands.
    continuum = (left_f + right_f) / 2.0

    # 2. Subtract the continuum from the peak to isolate the spike signal.
    spike_intensity = peak_f - continuum

    # 3. Threshold the result to get an initial mask.
    _, initial_mask = cv2.threshold(
        spike_intensity, spike_thresh, 255, cv2.THRESH_BINARY)
    initial_mask = initial_mask.astype(np.uint8)

    # 4. Use a two-step morphological process to clean the final mask.
    small_kernel = np.ones((3, 3), np.uint8)
    large_kernel = np.ones((7, 7), np.uint8)
    noise_removed = cv2.morphologyEx(
        initial_mask, cv2.MORPH_OPEN, small_kernel)
    final_mask = cv2.morphologyEx(noise_removed, cv2.MORPH_CLOSE, large_kernel)

    return final_mask


# --- Main execution block ---
if __name__ == "__main__":
    # --- IMPORTANT ---
    # Change these paths to where you saved your three images.
    path_769nm = "img-proc-4-wildfire/769nm-cam.jpg"  # The peak
    path_760nm = "img-proc-4-wildfire/760nm-cam.jpg"  # The left shoulder
    path_779nm = "img-proc-4-wildfire/779nm-cam.jpg"  # The right shoulder

    try:
        img_769nm = cv2.imread(path_769nm, cv2.IMREAD_GRAYSCALE)
        img_760nm = cv2.imread(path_760nm, cv2.IMREAD_GRAYSCALE)
        img_779nm = cv2.imread(path_779nm, cv2.IMREAD_GRAYSCALE)
        if any(img is None for img in [img_769nm, img_760nm, img_779nm]):
            raise FileNotFoundError("One or more image files were not found.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit()

    print("Aligning 'shoulder' images to the 'peak' image...")
    img_760nm_aligned = align_images(img_769nm, img_760nm)
    img_779nm_aligned = align_images(img_769nm, img_779nm)
    print("Alignment complete.")

    final_mask = calculate_continuum_removal(
        img_769nm, img_760nm_aligned, img_779nm_aligned)

    # --- Display logic for 4 panels ---
    display_size = (400, 400)
    img_peak_display = cv2.resize(img_769nm, display_size)
    img_left_display = cv2.resize(img_760nm_aligned, display_size)
    img_right_display = cv2.resize(img_779nm_aligned, display_size)
    mask_display = cv2.resize(final_mask, display_size)

    img_peak_display = cv2.cvtColor(img_peak_display, cv2.COLOR_GRAY2BGR)
    img_left_display = cv2.cvtColor(img_left_display, cv2.COLOR_GRAY2BGR)
    img_right_display = cv2.cvtColor(img_right_display, cv2.COLOR_GRAY2BGR)
    mask_display = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)

    cv2.putText(img_peak_display, '769nm (Peak)', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img_left_display, '760nm (Left)', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img_right_display, '779nm (Right)', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(mask_display, 'Final Output', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    top_row = np.hstack((img_peak_display, img_left_display))
    bottom_row = np.hstack((img_right_display, mask_display))
    combined_view = np.vstack((top_row, bottom_row))

    cv2.imshow('Three-Band Continuum Removal', combined_view)
    print("Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
