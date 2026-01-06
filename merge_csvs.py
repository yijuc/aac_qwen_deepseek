import csv
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('tp8_csv', help='benchmark_tp8_results.csv')
parser.add_argument('tp4_csv', help='benchmark_tp4_results.csv')
parser.add_argument('-o', '--output', help='output csv file (default: same folder as tp8_csv)')
args = parser.parse_args()

def read_csv(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    return rows

tp8_rows = read_csv(args.tp8_csv)
tp4_rows = read_csv(args.tp4_csv)

header = tp8_rows[0]
# 在 Mean_TTFT_ms 和 Mean_TPOT_ms 後各插入一個空欄
def insert_blanks(row):
    # 找到 TTFT 和 TPOT 欄位索引
    ttft_idx = header.index('Mean_TTFT_ms') + 1
    tpot_idx = header.index('Mean_TPOT_ms') + 2  # +1 for previous blank
    row = row[:ttft_idx] + [''] + row[ttft_idx:]
    row = row[:tpot_idx] + [''] + row[tpot_idx:]
    return row

# 合併，tp8/tp4交錯
merged = []
merged.append(header[:])  # header
merged[0] = insert_blanks(merged[0])

for i in range(1, max(len(tp8_rows), len(tp4_rows))):
    if i < len(tp8_rows):
        merged.append(insert_blanks(tp8_rows[i]))
    if i < len(tp4_rows):
        merged.append(insert_blanks(tp4_rows[i]))

if args.output:
    output_path = args.output
else:
    base = os.path.splitext(os.path.basename(args.tp8_csv))[0]
    output_path = os.path.join(os.path.dirname(args.tp8_csv), f"{base}_merge_result.csv")

with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(merged)

print(f"Done. Output: {output_path}")