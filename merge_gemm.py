import os
import pandas as pd
import argparse

def merge_gemm_csvs(input_folder):
    # 確保路徑是絕對路徑
    input_folder = os.path.abspath(input_folder)
    
    if not os.path.exists(input_folder):
        print(f"❌ 錯誤: 找不到路徑 {input_folder}")
        return

    # 1. 搜尋所有以 untuned_gemm.csv 結尾的檔案
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('untuned_gemm.csv')]
    
    if not csv_files:
        print(f"⚠️ 在 {input_folder} 找不到任何 *untuned_gemm.csv 檔案")
        return

    print(f"🔍 找到 {len(csv_files)} 個 CSV 檔案，準備合併...")

    all_dfs = []
    for file in csv_files:
        file_path = os.path.join(input_folder, file)
        try:
            # 讀取 CSV，確保 M, N, K 是整數
            df = pd.read_csv(file_path)
            # 統一欄位名稱為大寫（防止部分檔案寫成 m,n,k）
            df.columns = [c.upper() for c in df.columns]
            all_dfs.append(df)
            print(f"   - 已讀取: {file}")
        except Exception as e:
            print(f"   - 讀取 {file} 失敗: {e}")

    # 2. 合併所有 DataFrame
    merged_df = pd.concat(all_dfs, ignore_index=True)

    # 3. 根據 M, N, K 進行去重
    # sort_values 讓輸出的 CSV 更整齊
    final_df = merged_df.drop_duplicates(subset=['M', 'N', 'K']).sort_values(by=['M', 'N', 'K'])

    # 4. 輸出檔案
    output_path = os.path.join(input_folder, "merge_untunued_gemm.csv")
    final_df.to_csv(output_path, index=False)

    print(f"\n✨ 合併完成！")
    print(f"✅ 總筆數 (去重後): {len(final_df)}")
    print(f"💾 儲存路徑: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合併目錄下所有 *untuned_gemm.csv 並去重")
    parser.add_argument("-i", "--input", required=True, help="Input 資料夾路徑")
    args = parser.parse_args()

    merge_gemm_csvs(args.input)