import os
import json
import csv
import argparse
from datetime import datetime

def parse_logs(input_dir, output_file):
    results = []
    
    # Retrieve .json files from the directory
    log_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not log_files:
        print(f"No .json files found in {input_dir}")
        return

    for file_name in log_files:
        file_path = os.path.join(input_dir, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            config = data.get("config", {})
            perf = data.get("perf_result", {})
            
            if not perf:
                continue

            # Attempt to extract timestamp from filename (e.g., 1768931597)
            # fall back to file modification time on failure
            try:
                # Filename format typically ends with timestamp_ai_perf_benchmark.json
                ts_part = file_name.split('_')[-3] 
                timestamp = datetime.fromtimestamp(int(ts_part)).strftime('%Y-%m-%d %H:%M:%S')
            except:
                timestamp = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')

            # Construct data row including TP in the requested order
            row = {
                "TimeStamp": timestamp,
                "JsonFlieName": file_name,
                "TP": config.get("tp", "N/A"), # Extracting TP from config
                "MIN_Input": config.get("min_input_len", "N/A"),
                "MAX_Input": config.get("max_input_len", "N/A"),
                "MIN_Output": config.get("min_output_len", "N/A"),
                "MAX_Output": config.get("max_output_len", "N/A"),
                "Num_Prompts": config.get("query_num", "N/A"),
                "Generate_Token_Throughput": perf.get("generate_tokens_per_second(tps)", "N/A"),
                "Total_Token_Throughput": perf.get("total_tokens_per_second(tps)", "N/A"),
                "Request_throughput_req_s": perf.get("queries_per_second", "N/A"),
                "Concurrency": perf.get("concurrency", "N/A"),
                "Mean_TTFT_ms": perf.get("average_prefill_time", "N/A"),
                "Mean_TPOT_ms": perf.get("average_decode_time", "N/A")
            }
            results.append(row)
            
        except Exception as e:
            print(f"Error parsing file {file_name}: {e}")

    if not results:
        print("No successful results to export.")
        return

    # Define column order: TP is now the 3rd column
    headers = [
        "TimeStamp", "JsonFlieName", "TP", "MIN_Input", "MAX_Input", 
        "MIN_Output", "MAX_Output", "Num_Prompts", 
        "Generate_Token_Throughput", "Total_Token_Throughput", 
        "Request_throughput_req_s", "Concurrency", 
        "Mean_TTFT_ms", "Mean_TPOT_ms"
    ]

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        # Write header row
        writer.writeheader()
        for data in results:
            writer.writerow(data)

    print(f"Successfully parsed {len(results)} JSON reports. Summary saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse Kunlun JSON Benchmark logs and generate a CSV summary with TP field."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input json directory")
    parser.add_argument("-o", "--output", help="Path to output CSV file")
    
    args = parser.parse_args()
    input_path = os.path.abspath(args.input)
    output_path = args.output if args.output else os.path.join(input_path, "benchmark_json_summary.csv")
    
    parse_logs(input_path, output_path)