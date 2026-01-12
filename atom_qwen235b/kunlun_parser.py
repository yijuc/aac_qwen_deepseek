import os
import json
import csv
import argparse

def parse_json_perf(input_dir, output_file):
    results = []
    
    # 取得目錄下所有 .json 檔案
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"❌ 找不到任何 .json 檔案在 {input_dir}")
        return

    print(f"🔍 開始掃描目錄: {input_dir}")

    for file_name in json_files:
        file_path = os.path.join(input_dir, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 檢查是否有意義 (包含 perf_result 欄位)
                if "perf_result" not in data:
                    continue
                
                config = data.get("config", {})
                perf = data.get("perf_result", {})
                
                # 提取指定欄位
                row = {
                    "log file name": file_name,
                    # 新增 config 欄位
                    "min_input_len": config.get("min_input_len", "N/A"),
                    "max_input_len": config.get("max_input_len", "N/A"),
                    "min_output_len": config.get("min_output_len", "N/A"),
                    "max_output_len": config.get("max_output_len", "N/A"),
                    "tp": config.get("tp", "N/A"),
                    # 原有效能欄位
                    "generate tokens/s": perf.get("generate_tokens_per_second(tps)", "N/A"),
                    "total throughput tokens/s": perf.get("total_tokens_per_second(tps)", "N/A"),
                    "qps": perf.get("queries_per_second", "N/A"),
                    "concurrency": perf.get("concurrency", config.get("concurrency", "N/A")),
                    "avg prefill time": perf.get("average_prefill_time", "N/A"),
                    "avg decode time": perf.get("average_decode_time", "N/A")
                }
                results.append(row)
                print(f" ✅ 成功解析: {file_name}")
                
        except Exception as e:
            print(f" ⚠️ 無法處理檔案 {file_name}: {e}")

    if not results:
        print("❌ 沒有找到包含 'perf_result' 的有效 JSON 檔案。")
        return

    # CSV 欄位順序 (依照您要求的順序排列)
    headers = [
        "log file name",
        "min_input_len",
        "max_input_len",
        "min_output_len",
        "max_output_len",
        "tp",
        "generate tokens/s",
        "total throughput tokens/s",
        "qps",
        "concurrency",
        "avg prefill time",
        "avg decode time"
    ]

    # 寫入 CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✨ 解析完成！總計 {len(results)} 筆數據。")
    print(f"📝 匯總檔案位於: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="解析效能測試 JSON 檔案並匯總至 CSV (包含 Config 資訊)")
    parser.add_argument("-i", "--input", required=True, help="輸入 JSON 資料夾路徑")
    parser.add_argument("-o", "--output", help="輸出 CSV 檔案路徑 (預設為 summary.csv)")
    
    args = parser.parse_args()
    
    input_path = os.path.abspath(args.input)
    output_path = args.output if args.output else os.path.join(input_path, "summary.csv")
    
    parse_json_perf(input_path, output_path)