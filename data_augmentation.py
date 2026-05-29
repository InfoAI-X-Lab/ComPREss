import pandas as pd
import random
import os
from collections import Counter
REPLACEABLE_AA = ['G', 'V', 'L', 'I', 'S', 'T']
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df
def get_class_distribution(df):
    return Counter(df['label'])

def generate_augmented_sequence(sequence, seed=None):
    if seed is not None:
        random.seed(seed)
    seq_len = len(sequence)
    front_range = list(range(1, min(16, seq_len)))
    end_range = list(range(max(0, seq_len - 15), seq_len))
    replaceable_positions = front_range + end_range
    valid_positions = []
    for pos in replaceable_positions:
        if sequence[pos] in REPLACEABLE_AA:
            valid_positions.append(pos)
    if not valid_positions:
        return None
    replace_pos = random.choice(valid_positions)
    new_sequence = sequence[:replace_pos] + 'A' + sequence[replace_pos+1:]
    return new_sequence

def augment_class_data(df, target_class, target_count):
    class_df = df[df['label'] == target_class].copy()
    original_count = len(class_df)
    if original_count >= target_count:
        print(f"Class {target_class} : Original count {original_count} already meets or exceeds target {target_count}. Skipping augmentation.")
        return df
    print(f"Class {target_class} : Original count is {original_count}，augmenting to target {target_count}")
    augmented_sequences = set()
    original_sequences = set(class_df['sequence'])
    for _, row in class_df.iterrows():
        sequence = row['sequence']
        attempts = 0
        max_attempts = 100
        while attempts < max_attempts:
            augmented_seq = generate_augmented_sequence(sequence, seed=None)
            if augmented_seq is None:
                break
            if augmented_seq not in augmented_sequences and augmented_seq not in original_sequences:
                augmented_sequences.add(augmented_seq)
            attempts += 1
            if len(augmented_sequences) + original_count >= target_count:
                break
    needed_count = target_count - original_count
    final_augmented_sequences = list(augmented_sequences)
    if len(final_augmented_sequences) < needed_count:
        print(f"Warning: Only {len(final_augmented_sequences)} unique augmented sequences generated for Class {target_class}. Performing oversampling with replacement.")
        additional_needed = needed_count - len(final_augmented_sequences)
        sampled_sequences = random.choices(final_augmented_sequences, k=additional_needed)
        final_augmented_sequences.extend(sampled_sequences)
    else:
        final_augmented_sequences = random.sample(final_augmented_sequences, needed_count)
    augmented_df = pd.DataFrame({
        'label': [target_class] * len(final_augmented_sequences),
        'sequence': final_augmented_sequences
    })
    result_df = pd.concat([df, augmented_df], ignore_index=True)
    print(f"Class {target_class}: Augmentation completed. Added {len(final_augmented_sequences)} sequences.")
    return result_df

def main():
    # 设置输入和输出文件路径
    input_file = 'train.csv'
    output_file = 'train_augmented.csv'
    df = load_data(input_file)
    class_distribution = get_class_distribution(df)
    print("Original Class Distribution:")
    for class_label, count in class_distribution.items():
        print(f"  Class {class_label}: {count} sequences")
    target_count = 100
    augmented_df = df.copy()
    for class_label in class_distribution.keys():
        augmented_df = augment_class_data(augmented_df, class_label, target_count)
    augmented_distribution = get_class_distribution(augmented_df)
    print("\nAugmented Class Distribution:")
    for class_label, count in augmented_distribution.items():
        print(f"  Class {class_label}: {count} sequences")
    augmented_df.to_csv(output_file, index=False)
    print(f"\nSuccess! Augmented dataset saved to: {output_file}")

if __name__ == "__main__":
    random.seed(42)
    main()