
def is_header(line):
    words = set(line.lower().split())
    if len(words) <= 3:
        mapping = {"volume", "chapter",}
        return words & mapping
    else:
        False


lines = []
with open("tokenizer-gpt/austen-emma.txt", "r", encoding='utf-8') as f:
    lines = [line for line in f.readlines() if line.strip() != '']

#######################################################################

merged_lines = []
i = 0

while i < len(lines) - 1:

    first_line = lines[i].rstrip()
    second_line = lines[i + 1].rstrip()


    if is_header(first_line):
        merged_lines.append(first_line)
        i += 1
    else:
        if is_header(second_line):
            merged_lines.append(first_line)
            merged_lines.append(second_line)
            i += 2
        else:
            merged_lines.append(first_line + " " + second_line)
            i += 2


if i < len(lines):
    merged_lines.append(lines[i].rstrip())

# Сохраняем исправленный результат
corrected_output_path = "tokenizer-gpt/processed-austen-emma.txt"
with open(corrected_output_path, "w", encoding="utf-8") as file:
    file.write("\n".join(merged_lines))
