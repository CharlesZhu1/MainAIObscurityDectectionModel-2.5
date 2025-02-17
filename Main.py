import os
import csv
import re
import math
from collections import Counter
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(
    api_key="put your own"
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

        # Remove the 10 lowest outliers before calculating average obscurity for the generated essay
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
    if abs(average_obscurity_input - average_obscurity_generated_essays) / average_obscurity_input <= 0.12:
        print("\nResult: AI")
    else:
        print("\nResult: HUMAN")

if __name__ == "__main__":
    main()
