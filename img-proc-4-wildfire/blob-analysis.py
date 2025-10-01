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


def analyze_blobs(img_769, img_779, brightness_thresh=230, ratio_thresh=4.2, min_area=1):
    """
    Finds bright blobs (contours) and classifies them based on their
    average intensity ratio between the two images.
    """
    # 1. Find all bright regions in the 769nm image to serve as candidates.
    _, initial_mask = cv2.threshold(
        img_769, brightness_thresh, 255, cv2.THRESH_BINARY)

    # 2. Find the distinct contours (outlines) of each blob.
    contours, _ = cv2.findContours(
        initial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fire_contours = []
    epsilon = 1e-7  # To prevent division by zero

    # 3. Analyze each blob (contour).
    for contour in contours:
        # Ignore very small contours that are likely noise.
        if cv2.contourArea(contour) < min_area:
            continue

        # Create a mask for the current contour to calculate the mean intensity within it.
        mask = np.zeros(img_769.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Calculate the average pixel intensity within the contour for both images.
        avg_769 = cv2.mean(img_769, mask=mask)[0]
        avg_779 = cv2.mean(img_779, mask=mask)[0]

        # 4. Classify the blob based on the ratio of the *average* intensities.
        if (avg_769 / (avg_779 + epsilon)) > ratio_thresh:
            fire_contours.append(contour)

    # Create the final output mask by drawing the confirmed fire contours.
    final_mask = np.zeros(img_769.shape, dtype=np.uint8)
    cv2.drawContours(final_mask, fire_contours, -1, 255, thickness=cv2.FILLED)

    return final_mask


# --- Main execution block ---
if __name__ == "__main__":
    path_769nm = "img-proc-4-wildfire/769nm.png"
    path_779nm = "img-proc-4-wildfire/780nm.png"

    try:
        img_769nm = cv2.imread(path_769nm, cv2.IMREAD_GRAYSCALE)
        img_779nm = cv2.imread(path_779nm, cv2.IMREAD_GRAYSCALE)
        if img_769nm is None or img_779nm is None:
            raise FileNotFoundError("One or both image files were not found.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit()

    print("Aligning images...")
    img_779nm_aligned = align_images(img_769nm, img_779nm)
    print("Alignment complete.")

    final_mask = analyze_blobs(img_769nm, img_779nm_aligned)

    # --- Display logic ---
    display_size = (500, 500)
    img_769_display = cv2.resize(img_769nm, display_size)
    img_779_display = cv2.resize(img_779nm_aligned, display_size)
    mask_display = cv2.resize(final_mask, display_size)

    img_769_display = cv2.cvtColor(img_769_display, cv2.COLOR_GRAY2BGR)
    img_779_display = cv2.cvtColor(img_779_display, cv2.COLOR_GRAY2BGR)
    mask_display = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)

    cv2.putText(img_769_display, '769nm Input', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img_779_display, '779nm Aligned', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(mask_display, 'Blob Analysis Output', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    combined_view = np.hstack((img_769_display, img_779_display, mask_display))

    cv2.imshow('Wildfire Detection (Blob Analysis)', combined_view)
    print("Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
