"""Enable CCD for the remaining untangling steps once few contours remain."""

CCD_CONTOUR_THRESHOLD = 16


def enable_ccd_for_few_contours(solver):
    """Check after a completed physical step; never disable CCD again.

    Read actual contour data rather than optional PRP debug statistics. Missing
    counts are not evidence of zero intersections. Skip animation-only phases
    where untangling is disabled, and avoid fetching data once CCD is enabled.
    """
    config = solver.get_config()
    if config.use_ccd_linesearch or not config.use_untangling:
        return False

    counts = solver.get_intersection_contour_data().get("num_contours")
    if counts is None or len(counts) != 1:
        return False
    count = int(counts[0])
    if not 0 <= count < CCD_CONTOUR_THRESHOLD:
        return False

    config.use_ccd_linesearch = True
    print(
        f"[Auto CCD] Frame {config.current_frame}: {count} contours < "
        f"{CCD_CONTOUR_THRESHOLD}; enabling use_ccd_linesearch for subsequent steps.",
        flush=True)
    return True
