import math
import random
from typing import List, Dict, Any
import json

# Set a random seed for reproducibility
random.seed(42)

def calculate_attentional_weights(current_index: int, total_words: int, window_size: int = 3, decay: float = 0.5) -> List[float]:
    """
    Calculate attentional weights for each word based on their distance from the current fixation.

    Parameters:
    - current_index: Index of the currently fixated word.
    - total_words: Total number of words in the stimulus.
    - window_size: Number of words to consider on either side of the current fixation.
    - decay: Decay rate for attentional weights.

    Returns:
    - List of attentional weights for each word.
    """
    weights = [0.0] * total_words
    for offset in range(-window_size, window_size + 1):
        target = current_index + offset
        if 0 <= target < total_words:
            distance = abs(offset)
            weight = math.exp(-decay * distance)
            weights[target] = weight
    # Normalize weights
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    return weights

def calculate_fix_duration(word: Dict[str, Any],
                           attentional_weight: float,
                           base_time: float = 100.0,
                           a: float = 10.0,
                           b: float = 50.0,
                           c: float = 5.0) -> float:
    """
    Calculate fixation duration based on word attributes and attentional weight.

    Parameters:
    - word: Dictionary containing word attributes.
    - attentional_weight: Attentional weight assigned to the word.
    - base_time: Base fixation duration in ms.
    - a, b, c: Coefficients for frequency, predictability, and integration_time.

    Returns:
    - fix_duration in milliseconds.
    """
    frequency = word.get('frequency', 1.0)
    predictability = word.get('predictability', 0.0)
    integration_time = word.get('integration_time', 0.0)
    integration_failure = word.get('integration_failure', 0.0)

    # Avoid log(0) by ensuring frequency is at least 1
    frequency = max(frequency, 1.0)

    # Calculate components
    freq_component = a * math.log(frequency)
    predict_component = b * (1 - predictability)
    integration_component = c * integration_time

    # Random noise influenced by integration_failure
    # Higher integration_failure leads to higher variability
    noise = random.gauss(0, 1) * (1 + integration_failure * 10)

    # Total fixation duration adjusted by attentional weight
    fix_duration = (base_time + freq_component + predict_component + integration_component) / attentional_weight + noise

    # Ensure fixation duration is positive
    fix_duration = max(fix_duration, 50.0)  # Minimum fixation duration of 50 ms

    return fix_duration

def decide_next_fixation(attentional_weights: List[float], current_index: int, regression_prob: float = 0.1) -> int:
    """
    Decide the next fixation target based on attentional weights and regression probability.

    Parameters:
    - attentional_weights: List of attentional weights for each word.
    - current_index: Index of the currently fixated word.
    - regression_prob: Probability of making a regression (backward saccade).

    Returns:
    - Index of the next fixation word.
    """
    if random.random() < regression_prob and current_index > 0:
        # Make a regression: move back by 1 word
        return current_index - 1

    # Otherwise, choose next fixation based on attentional weights to the right
    right_weights = attentional_weights[current_index:]
    total_right_weight = sum(right_weights)
    if total_right_weight == 0:
        return current_index  # Stay on the current word if no weight to the right

    # Normalize right weights
    normalized_weights = [w / total_right_weight for w in right_weights]

    # Choose next fixation based on normalized weights
    next_fix = random.choices(range(current_index, len(attentional_weights)), weights=normalized_weights, k=1)[0]
    return next_fix

def process_stimuli(input_data: List[Dict[str, Any]],
                    window_size: int = 3,
                    decay: float = 0.5,
                    regression_prob: float = 0.1) -> List[Dict[str, Any]]:
    """
    Process input stimuli and generate fixation data sequences.

    Parameters:
    - input_data: List of stimuli dictionaries.
    - window_size: Number of words to consider on either side for attentional weights.
    - decay: Decay rate for attentional weights.
    - regression_prob: Probability of making a regression.

    Returns:
    - List of stimuli with fixation sequences.
    """
    output_data = []

    for stimulus in input_data:
        fixation_sequence = []
        words = stimulus.get('words', [])
        total_words = len(words)
        if total_words == 0:
            # Skip stimuli with no words
            continue

        # Initialize reading sequence
        current_index = 0  # Start at the first word
        end_of_text = False
        max_fixations = total_words * 2  # Arbitrary limit to prevent infinite loops
        fix_count = 0

        while not end_of_text and fix_count < max_fixations:
            fix_count += 1
            attentional_weights = calculate_attentional_weights(current_index, total_words, window_size, decay)
            current_word = words[current_index]
            attentional_weight = attentional_weights[current_index]
            fix_duration = calculate_fix_duration(current_word, attentional_weight)

            fixation = {
                "fix_x": None,
                "fix_y": None,
                "norm_fix_x": None,
                "norm_fix_y": None,
                "fix_duration": fix_duration,
                "word_index": current_index
            }
            fixation_sequence.append(fixation)

            # Decide next fixation
            next_index = decide_next_fixation(attentional_weights, current_index, regression_prob)

            if next_index == current_index:
                # If staying on the same word, consider moving forward to prevent infinite loop
                next_index = current_index + 1

            if next_index >= total_words:
                end_of_text = True
            else:
                current_index = next_index

        output_stimulus = {
            "stimulus_index": stimulus.get("stimulus_index"),
            "participant_index": stimulus.get("participant_index"),
            "time_constraint": stimulus.get("time_constraint"),
            "baseline_model_name": stimulus.get("baseline_model_name"),
            "fixation_data": fixation_sequence
        }

        output_data.append(output_stimulus)

    return output_data

# Example usage
if __name__ == "__main__":
    
    # Load input data from a file
    with open("swift_input_data.json", "r") as f:
        input_data = json.load(f)

    # Process stimuli
    output_data = process_stimuli(input_data)

    # # Print output
    # print(json.dumps(output_data, indent=4))
    
    # Save output to a file
    with open("swift_output_data.json", "w") as f:
        json.dump(output_data, f, indent=4)
