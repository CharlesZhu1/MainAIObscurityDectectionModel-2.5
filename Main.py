import os
import csv
import re
import math
from collections import Counter
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(
    api_key="Random one"
    # Replace with your actual API key
)

def load_global_frequencies_csv(file_name):
    """Load global word frequencies from a CSV file."""
    frequencies = {}
    total_words_in_global = 0

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                word, freq = row
                frequencies[word.lower()] = float(freq)
                total_words_in_global += float(freq)
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found at path: {file_path}")
    except Exception as e:
        print(f"Error reading file '{file_name}': {e}")

    return frequencies, total_words_in_global

def calculate_obscurity(word, global_frequencies, total_words_in_global, essay_word_count):
    """Calculate the obscurity for a single word."""
    if word not in global_frequencies:
        return None  # Skip words not in the global frequencies

    # ChatGPT's usage: 1 / total words in the essay
    chatgpt_usage = 1 / essay_word_count
    
    # Global usage: frequency of the word / total words in the global corpus
    global_usage = global_frequencies[word] / total_words_in_global

    # Calculate obscurity (log ratio)
    obscurity = math.log(chatgpt_usage / global_usage, 2)
    return obscurity

def remove_lowest_outliers(obscurities, count=10):
    """Remove the N lowest obscurity outliers."""
    sorted_obscurities = sorted(obscurities, key=lambda x: x[1])  # Sort by obscurity values
    return sorted_obscurities[count:]  # Remove the first 'count' elements

def generate_essay(essay_word_count, thesis):
    """Generate an essay using OpenAI with the same thesis and roughly the same word count."""
    prompt = f"Generate an essay based on the thesis: '{thesis}'. The essay should be approximately {essay_word_count} words long."
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Please generate an essay."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

def main():
    print("Calculating obscurity for an input essay...\n")
    print("Enter your essay (press Ctrl+D to finish):")

    # Load global word frequencies
    global_word_frequencies, total_words_in_global = load_global_frequencies_csv('unigram_freq.csv')

    if not global_word_frequencies:
        print("Failed to load global word frequencies. Exiting...")
        return

    # Take essay input from the user
    essay = []
    try:
        while True:
            line = input()  # Collect the essay line by line
            essay.append(line)
    except EOFError:
        essay = ' '.join(essay)

    # Count words in the essay
    words = re.findall(r'\b\w+\b', essay.lower())  # Extract words
    essay_word_count = len(words)

    # Calculate obscurity for each word, including repetitions
    word_obscurities = []
    for word in words:  # Loop through each individual word, including repetitions
        obscurity = calculate_obscurity(word, global_word_frequencies, total_words_in_global, essay_word_count)
        if obscurity is not None:
            word_obscurities.append((word, obscurity))

    # Display the individual words and their obscurity scores
    print("\nWord Obscurities for the input essay:")
    for word, obscurity in word_obscurities:
        print(f"{word}: {obscurity:.4f}")

    # Remove the 10 lowest outliers before calculating average obscurity
    filtered_word_obscurities = remove_lowest_outliers(word_obscurities, count=10)

    # Calculate the average obscurity score for the input essay
    if filtered_word_obscurities:
        average_obscurity_input = sum([obscurity for _, obscurity in filtered_word_obscurities]) / len(filtered_word_obscurities)
    else:
        average_obscurity_input = 0

    print(f"\nAverage Obscurity for the input essay (after removing outliers): {average_obscurity_input:.4f}")

    # Generate 3 essays based on the same thesis and roughly the same word count
    thesis = " ".join(essay.split('.')[:2])  # Get the first two sentences to form the thesis
    generated_essays = []
    for _ in range(3):  # Generate 3 essays
        generated_essay = generate_essay(essay_word_count, thesis)
        generated_essays.append(generated_essay)

    # Calculate the obscurity for each generated essay and compute the average
    average_obscurity_generated_essays = 0
    for generated_essay in generated_essays:
        # Count words in the generated essay
        words_generated = re.findall(r'\b\w+\b', generated_essay.lower())
        essay_word_count_generated = len(words_generated)

        # Calculate obscurity for each word in the generated essay
        word_obscurities_generated = []
        for word in words_generated:
            obscurity = calculate_obscurity(word, global_word_frequencies, total_words_in_global, essay_word_count_generated)
            if obscurity is not None:
                word_obscurities_generated.append((word, obscurity))

        # Remove the 10 lowest outliers before calculating average obscurity
        filtered_word_obscurities_generated = remove_lowest_outliers(word_obscurities_generated, count=10)

        # Calculate the average obscurity score for the generated essay
        if filtered_word_obscurities_generated:
            average_obscurity_generated = sum([obscurity for _, obscurity in filtered_word_obscurities_generated]) / len(filtered_word_obscurities_generated)
        else:
            average_obscurity_generated = 0

        average_obscurity_generated_essays += average_obscurity_generated

    # Average obscurity score for the 3 generated essays
    average_obscurity_generated_essays /= 3

    print(f"\nAverage Obscurity for the 3 generated essays (after removing outliers): {average_obscurity_generated_essays:.4f}")

    # Compare the two average obscurity scores
    if abs(average_obscurity_input - average_obscurity_generated_essays) / average_obscurity_input <= 0.14:
        print("\nResult: AI")
    else:
        print("\nResult: HUMAN")

    # ----------------------------------------------------------------------------
    # ADDED SECTION: Confidence score system using a normal distribution
    # ----------------------------------------------------------------------------
    # We define "difference" as how far off the essay's obscurity is from the AI-generated mean.
    # We assume a normal distribution with mean=0 and std dev=4.
    # Then compute the standard normal CDF to get a confidence measure in the range [0,1].
    # A higher value here implies we are "less sure" it's AI (per your request).

    difference = average_obscurity_input - average_obscurity_generated_essays
    # Standard deviation = 4
    z_score = difference / 4.0

    # Standard Normal CDF using error function:
    # CDF(z) = 0.5 * (1 + erf(z / sqrt(2)))
    cdf_value = 0.5 * (1 + math.erf(z_score / math.sqrt(2)))

    # Convert to percentage
    confidence_percent = cdf_value * 100.0

    print(f"\nConfidence Score (0-100): {confidence_percent:.2f}")
    print("   (Higher = Less sure it’s AI, based on empirical rule with std dev = 4)")
    # ----------------------------------------------------------------------------
   # ----------------------------------------------------------------------------
    # NEW SECTION: Heuristic-based AI detection (repetition and list detection)
    # ----------------------------------------------------------------------------
    print("\nRunning repetition and list-based analysis...")

    style_prompt = (
    "Analyze the human claimed text inputted. Does it show signs of repetitiveness or excessive listing "
    "that are typical of AI-generated writing? Specifically, check for (1) repeated phrases or patterns, "
    "(2) heavy use of enumeration or bullet-like structures, and (3) lack of natural flow. "
    "Answer only with 'Likely AI', 'Likely Human', or 'Unclear'.\n\n"
    f"Essay:\n{essay}"
)


    style_check = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a style analysis engine."},
            {"role": "user", "content": style_prompt}
        ]
    )
    style_judgment = style_check.choices[0].message.content.strip()

    # Assign style-based confidence
    if "Likely AI" in style_judgment:
        style_confidence = 0  # Confident it's AI
    elif "Likely Human" in style_judgment:
        style_confidence = 100  # Confident it's human
    else:
        style_confidence = 50  # Unclear

    print(f"Style-Based Report: {style_judgment}")
    print(f"Style-Based Confidence Score (0-100): {style_confidence:.2f}")

    # ----------------------------------------------------------------------------
    # COMBINE FINAL CONFIDENCE: 75% obscurity-based, 25% style-based
    # ----------------------------------------------------------------------------
    final_confidence_score = (0.75 * confidence_percent) + (0.25 * style_confidence)
    print(f"\n🔎 Final Combined Confidence Score (0-100): {final_confidence_score:.2f}")
    if final_confidence_score < 50:
        print("=> Final Verdict: Likely AI")
    elif final_confidence_score > 70:
        print("=> Final Verdict: Likely Human")
    else:
        print("=> Final Verdict: Unclear")

if __name__ == "__main__":
    main()
