import pandas as pd

def recommend_swap_window(df, fatigue_column="fatigued", min_consecutive=3):
    """
    Finds a window of laps where fatigue is consistently high.
    Args:
        df (pd.DataFrame): The lap data including a binary fatigue indicator
        fatigue_column (str): Name of the column indicating fatigue (1 = fatigued)
        min_consecutive (int): Minimum consecutive fatigued laps to trigger recommendation
    Returns:
        str or tuple: Recommended lap range or explanation
    """
    fatigued_laps = df[df[fatigue_column] == 1]["lap"].tolist()
    
    if not fatigued_laps:
        return "✅ No swap needed: driver shows no sustained fatigue."

    # Find consecutive laps
    groups = []
    group = [fatigued_laps[0]]

    for i in range(1, len(fatigued_laps)):
        if fatigued_laps[i] == fatigued_laps[i - 1] + 1:
            group.append(fatigued_laps[i])
        else:
            if len(group) >= min_consecutive:
                groups.append(group)
            group = [fatigued_laps[i]]
    if len(group) >= min_consecutive:
        groups.append(group)

    if groups:
        rec = groups[0]
        return f"⚠️ Recommend swap between laps {rec[0]}–{rec[-1]}"
    else:
        return "🟡 Fatigue detected, but not sustained enough to recommend swap yet."
