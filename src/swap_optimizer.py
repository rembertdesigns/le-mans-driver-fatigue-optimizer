def recommend_swap_window(fatigue_preds, threshold=0.7):
    """
    Returns lap indices where fatigue crosses threshold.
    """
    swap_laps = [i for i, pred in enumerate(fatigue_preds) if pred > threshold]
    if not swap_laps:
        return "No swap needed yet"
    return f"Recommended swap between laps {min(swap_laps)} and {max(swap_laps)}"