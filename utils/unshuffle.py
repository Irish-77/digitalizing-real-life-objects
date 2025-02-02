import csv

def main():
    # Read the model mappings.
    mappings = []
    with open("model_mappings.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mappings.append(row)
    num_mappings = len(mappings)

    unshuffled_rows = []

    # Open the survey results.
    with open("2025-02-02_results_RAW.csv", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        unshuffled_rows.append(header)

        # Process each survey response.
        for row in reader:
            new_row = [row[0]]
            answers = row[1:]
            for i in range(0, len(answers), 2):
                mapping_index = (i // 2) % num_mappings
                mapping = mappings[mapping_index]

                letter_shape = answers[i].strip()
                letter_texture = answers[i+1].strip() if i+1 < len(answers) else ""

                model_shape = mapping.get(letter_shape, letter_shape)
                model_texture = mapping.get(letter_texture, letter_texture)

                new_row.extend([model_shape, model_texture])
            unshuffled_rows.append(new_row)

    # Write the unshuffled results to a new CSV file with all fields quoted.
    with open("unshuffled_results.csv", "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerows(unshuffled_rows)

if __name__ == "__main__":
    main()
