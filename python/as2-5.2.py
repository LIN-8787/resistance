import json
import math
import os
import re
from PIL import Image, ImageEnhance
import pytesseract
import streamlit as st
import pandas as pd
from google import genai
from google.genai import Client

# ==============================================================================
# ⚙️ 1. 設定 Tesseract OCR 執行檔路徑 (必須在最頂端執行)
# ==============================================================================
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    st.error("❌ 系統找不到 Tesseract OCR 引擎！請確認是否已下載並安裝於 C:\\Program Files\\Tesseract-OCR\\")


# ==============================================================================
#  設定 Gemini API KEY (建議從 Google AI Studio 申請免費額度)
# ==============================================================================
# 您可以直接將 API Key 貼在下方引號中，或設定為環境變數
if "VLM_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["VLM_API_KEY"]
else:
    GEMINI_API_KEY = "PLEASE_SET_KEY_IN_STREAMLIT_CLOUD"


def get_gemini_client():
    return Client(api_key=GEMINI_API_KEY)


# ==============================================================================
#  獨立大模型視覺解析函數
# ==============================================================================
def scan_image_with_vlm(image):
    
    try:
        client = get_gemini_client()
        
        # 確保 prompt 的完整定義出現在這裡，不能被刪除！
        prompt = """
        你是一個專業的電子硬體工程師。請仔細分析這張電路圖（Schematic），並執行以下任務：
        1. 圖形符號優先識別：不要只依賴文字名字！請優先尋找圖紙上的「電阻電路符號」，包含以下兩種樣式：
           - 樣式 A（美規）：兩端有引線的「鋸齒狀線條（Zigzag line）」。
           - 樣式 B（歐規）：兩端有引線的「空心細長方形（Rectangle）」。
           只要符合這兩種圖形，不論其位號名字是用 R, PR, SR, ZR 還是任何自訂代號，都必須判定為電阻並提取出來。
        
        2. 辨識它們各自的電阻值（例如 10K, 1K），如果前面帶有 NL/ 請務必保留（如 NL/10K）。
        3. 辨識它們的額定最大功率（例如 1/16W, 1/4W），若圖中未特別標註，請合理推測為 1/16W。
        4. 關鍵任務：沿著電路圖的導線與邏輯關係，找出該電阻連接的「主要供電電壓軌」（例如 3.3V, 1.8V, 5V）。
        5. 不論是 PR、R、SR、ZR 還是任何命名開頭，
           只要符合大模型輸出的文字規格結構，通通提取內部數值代入 V²/R 公式計算。
        
        注意事項：
        - 徹底忽略 0402, 0603 等封裝尺寸代號。
        - 排除接腳編號（如 1, 2, 3, 4）。
        
        請嚴格以 JSON 陣列格式回傳，不要包含任何 Markdown 標籤 (如 ```json) 或額外的解釋文字。
        JSON 格式範例：
        [
          {"name": "PR328", "val": "10K", "p": "1/16W", "v": "3.3V"},
          {"name": "PR330", "val": "1K", "p": "1/16W", "v": "1.8V"}
        ]
        """
        
        # 將模型名稱確實升級為最新的 'gemini-2.5-flash'
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image, prompt] # 此處會完美撈取到上方定義好的 prompt
        )
        
        json_text = response.text.strip()
        json_text = re.sub(r'^```json\s*', '', json_text, flags=re.MULTILINE)
        json_text = re.sub(r'```$', '', json_text, flags=re.MULTILINE)
        
        parsed_data = json.loads(json_text)
        output_lines = []
        
        for item in parsed_data:
            if not isinstance(item, dict):
                continue
                
            r_name = (item.get("name") or item.get("id") or "").upper().strip()
            r_val = (item.get("val") or item.get("value") or "10K").upper().strip()
            r_power = (item.get("p") or item.get("power") or "1/16W").upper().strip()
            r_volt = (item.get("v") or item.get("voltage") or "3.3V").upper().strip()
            
            # 清洗與還原複雜電壓名稱
            r_val = r_val.replace(" ", "")
            r_volt = r_volt.replace(" ", "")
            
            if r_name:
                #只組裝電阻行格式，不再將電壓另外加入 voltages_set 容器中
                output_lines.append(f"{r_name}={r_val}_{r_power}_{r_volt}")
            
        # 直接對元件名單進行排序並以換行符號結合回傳
        # 這樣左側 Step 2 text_area 就絕對不會再出現任何前置的 V=+V3.3 等干擾行！
        if output_lines:
            return "\n".join(sorted(list(set(output_lines))))
        else:
            return "大模型未回傳有效元件數據"
        
    except Exception as e:
        return f"大模型視覺解析失敗。錯誤原因: {e}"
    
# ==============================================================================
#  讀取文字檔數值後計算
# ==============================================================================
def calculate_derating_metrics(user_text, derating_target=0.80):
    lines = user_text.split('\n')
    results = []
    fallback_voltage = 3.3

    for line in lines:
        line = line.strip()
        if not line or "=" not in line or line.upper().startswith("V="):
            continue

        name, spec = line.split('=', 1)
        name = name.strip().upper()
        clean_spec = spec.upper().strip()

        # 先用底線切開所有區塊
        parts = clean_spec.split('_')
        
        if len(parts) >= 3:
            # 修正 parts.replace 錯誤，精確指定 parts[0]
            raw_val = parts[0].replace("NL/", "").replace("1%", "").replace("5%", "").strip()
            
            # 防止 +V5_SB 被底線切碎的縫合機制
            # 如果最後一個區塊是 'SB'，代表完整的電壓名稱被切斷了，必須要把倒數兩個區塊重新接回來！
            if parts[-1].strip() == "SB" and len(parts) >= 4:
                raw_volt_name = "_".join(parts[-2:]).strip() # 重新接回 "+V5_SB"
                raw_power = "_".join(parts[1:-2]).strip()   # 中間留給功率
            else:
                raw_volt_name = parts[-1].strip()            # 正常情況：最後一個就是電壓
                raw_power = "_".join(parts[1:-1]).strip()    # 中間留給功率
            
            # ---- 💡 數值提取一：阻值換算純歐姆 (R) ----
            val_match = re.search(r'(\d+(?:\.\d+)?)\s*([KkMm𝛀𝛺R]?)', raw_val)
            if not val_match:
                continue
            val_num = float(val_match.group(1))
            unit = val_match.group(2)
            if 'K' in unit: val_num *= 1000
            elif 'M' in unit: val_num *= 1000000

            # ---- 💡 數值提取二：額定最大功率 (PMAX) ----
            p_max = 0.25 # 預設值
            # 支援 1/16W 或 1_16W 形式
            p_match_slash = re.search(r'(\d+)/(\d+)', raw_power)
            p_match_under = re.search(r'(\d+)_(\d+)', raw_power)
            
            if p_match_slash:
                p_max = float(p_match_slash.group(1)) / float(p_match_slash.group(2))
            elif p_match_under:
                p_max = float(p_match_under.group(1)) / float(p_match_under.group(2))
            else:
                p_num_match = re.search(r'(\d+(?:\.\d+)?)', raw_power)
                if p_num_match:
                    p_max = float(p_num_match.group(1))

            # ---- 💡 數值提取三：精準提取電壓 (V) ----
            # 此時 raw_volt_name 已被完美縫合，可精準辨識特規字串
            if "+V5_SB" in raw_volt_name:
                voltage_used = 5.0
            elif "+V3.3SB" in raw_volt_name or "+V3.3 SB" in raw_volt_name:
                voltage_used = 3.3
            else:
                # 普通電壓字串（如 5V, 3.3V）提取其中的數字
                v_num_match = re.search(r'(\d+(?:\.\d+)?)', raw_volt_name)
                if v_num_match:
                    voltage_used = float(v_num_match.group(1))
                else:
                    voltage_used = fallback_voltage

            # ---- 💡 核心降額公式計算 ----
            p_act = (voltage_used ** 2) / val_num
            stress_ratio = p_act / p_max
            is_pass = stress_ratio <= derating_target

            results.append({
                "name": name,
                "r_value": val_num,
                "voltage_name_raw": raw_volt_name,
                "voltage_used": voltage_used,
                "p_act": p_act,
                "p_max": p_max,
                "stress_ratio": stress_ratio,
                "is_pass": is_pass
            })

    return results


# ==============================================================================
#  獨立計算公式程式 (從文字行內精準抽離各自電壓進行 Pact 計算)
# ==============================================================================
def calculate_derating_metrics(user_text, derating_target=0.80):
    lines = user_text.split('\n')
    results = []

    # 1. 收集全域保底電壓
    global_voltages = []
    for line in lines:
        v_match = re.search(r'V\s*=\s*(\d+(?:\.\d+)?)', line, re.IGNORECASE)
        if v_match and "=" in line and line.strip().upper().startswith("V"):
            global_voltages.append(float(v_match.group(1)))
            
    fallback_voltage = global_voltages[-1] if global_voltages else 3.3

    # 2. 逐行解析每個電阻
    for line in lines:
        line = line.strip()
        if not line or "=" not in line or line.upper().startswith("V="):
            continue

        name, spec = line.split('=', 1)
        name = name.strip().upper()
        clean_spec = spec.upper().strip()

        # 💡 【核心優化 1】：將複雜字串依底線切開，最尾端的一定是電壓相關資訊
        parts = clean_spec.split('_')
        if len(parts) < 2:
            continue  # 格式不符則跳過

        # 永遠鎖定最後一個區塊作為電壓來源 (例如: "5V", "+V5_SB", "3.3V")
        # 如果最後一個是 "SB"，則把倒數兩個合併 (應對 +V5_SB 被切成 ['+V5', 'SB'] 的狀況)
        if parts[-1] == "SB" and len(parts) >= 3:
            raw_volt_part = "_".join(parts[-2:])
            # 剩餘的前面部分就是 阻值 與 功率
            remaining_spec = "_".join(parts[:-2])
        else:
            raw_volt_part = parts[-1]
            remaining_spec = "_".join(parts[:-1])

        # 🚀 精準提取電壓數字
        voltage_used = None
        v_num_match = re.search(r'(\d+(?:\.\d+)?)', raw_volt_part)
        if v_num_match:
            voltage_used = float(v_num_match.group(1))
        else:
            voltage_used = fallback_voltage

        # 主動抹除剩餘字串中的 NL/ 與 1%/5% 精度干擾
        remaining_spec = remaining_spec.replace("NL/", "").replace("1%", "").replace("5%", "").strip()

        # 💡 【核心優化 2】：精準提取功率 (PMAX) 
        p_max = 0.25  # 預設 1/4W
        
        # 支援 1/16W, 1/8W 或 1_16W 的形式
        p_match = re.search(r'(\d+)\s*/\s*(\d+)\s*[WW]?', remaining_spec)
        p_match_under = re.search(r'(\d+)\s*_\s*(\d+)\s*[WW]?', remaining_spec)
        
        if p_match:
            p_max = float(p_match.group(1)) / float(p_match.group(2))
            remaining_spec = re.sub(r'\d+\s*/\s*\d+\s*[WW]?', '', remaining_spec).strip()
        elif p_match_under:
            p_max = float(p_match_under.group(1)) / float(p_match_under.group(2))
            remaining_spec = re.sub(r'\d+\s*_\s*\d+\s*[WW]?', '', remaining_spec).strip()
        else:
            # 單純數字型功率如 0.1W, 1W
            p_num_match = re.search(r'(\d+(?:\.\d+)?)\s*[WW]', remaining_spec)
            if p_num_match:
                p_max = float(p_num_match.group(1))
                remaining_spec = re.sub(r'(\d+(?:\.\d+)?)\s*[WW]', '', remaining_spec).strip()

        # 💡 【核心優化 3】：提取阻值 (R)
        # 此時 remaining_spec 裡只剩下阻值字串了（例如 "100K" 或 "1K"）
        val_match = re.search(r'(\d+(?:\.\d+)?)\s*([KkMm𝛀𝛺R]?)', remaining_spec)
        if not val_match:
            continue
            
        val_num = float(val_match.group(1))
        unit = val_match.group(2)
        if 'K' in unit: val_num *= 1000
        elif 'M' in unit: val_num *= 1000000

        # 降額核心計算
        p_act = (voltage_used ** 2) / val_num
        stress_ratio = p_act / p_max
        is_pass = stress_ratio <= derating_target

        results.append({
            "name": name,
            "r_value": val_num,
            "voltage_used": voltage_used, 
            "p_act": p_act,
            "p_max": p_max,
            "stress_ratio": stress_ratio,
            "is_pass": is_pass
        })

    return results

# ==============================================================================
#  Streamlit 使用者網頁介面
# ==============================================================================
st.set_page_config(layout="wide", page_title="Derating")
st.title("Derating Check")

DERATING_TARGET = 0.80

if "realtime_user_text" not in st.session_state:
    st.session_state.realtime_user_text = ""

col1, col2 = st.columns(2)

with col1:
    st.header("Step 1：上傳電路圖")
    uploaded_file = st.file_uploader("請上傳或拖曳圖片...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="已讀取電路圖", use_container_width=True)
        
        st.header("Step 2：AI 視覺邏輯辨識結果")
        
        # 觸發 VLM 智慧掃描辨識
        if "clean_ocr_output" not in st.session_state or st.button("🚀 重新啟動圖片分析"):
            with st.spinner("🧠 正在進行線路推理中..."):
                # 掃描完成後，同步將結果更新到 OCR 備份與當前即時文字狀態中
                vlm_result = scan_image_with_vlm(image)
                st.session_state.clean_ocr_output = vlm_result
                st.session_state.realtime_user_text = vlm_result  # 同步初始化文字框內容
        
        # 讓使用者可以即時修改與刪除 (綁定 key 機制)
        st.text_area(
            "元件與電壓清單 (若有小誤差可手動修改)：",
            value=st.session_state.realtime_user_text,
            height=250,
            key="realtime_user_text"  # 🔑 加上固定 key，由 st.session_state 主導管理
        )

with col2:
    st.header("Step 3：Derating 分析")
    if uploaded_file and st.button("開始計算Derating判定", type="primary"):
        with st.spinner("正在讀取行內專專規格並執行計算..."):
            try:
                # 從安全初始化後的狀態中提取即時文字
                current_text = st.session_state.realtime_user_text
                
                # 呼叫已經除錯修正（零件陣列索引已修復）的計算核心
                report_card = calculate_derating_metrics(current_text, DERATING_TARGET)
                
                st.success("✅ 文字檔內部數值分流判定完成！")
                st.write("---")
                
                if not report_card:
                    st.warning("⚠️ 文字框內無有效的電阻元件格式，請確認格式（如：HR1=10K_1/16W_3.3V）。")
            
                # 尋找所有算好的結果
                for component in report_card:
                    st.subheader(f"🔍 元件： {component['name']}")
                    
                    # 抓取該元件在文字框內拆出的電壓
                    this_r_voltage = component['voltage_used']
                    
                    # 印出各自跳轉的電壓與數值
                    st.info(f"工作電壓： `{this_r_voltage:.1f} V`")
                    st.markdown(f"* **電阻值 (R)**： `{component['r_value']:.1f} Ω` ")
                    st.markdown(f"* **量測工作功耗 (Pact)**： `{component['p_act']:.6f} W` ")
                    st.markdown(f"* **額定最大功率 (Pmax)**： `{component['p_max']:.4f} W` ")
                    
                    # 算式呈現，完美各自跳轉代入數值
                    st.markdown(f"* **Pact/Pmax計算 (V² / R / Pmax)**：")
                    st.code(f"({this_r_voltage:.1f}V)² / {component['r_value']:.0f}Ω / {component['p_max']:.4f}W = {component['stress_ratio']:.4f}")
                    st.markdown(f"* ****： `{component['stress_ratio']*100:.1f}%` (Derating標準: {DERATING_TARGET*100}%)")
                    
                    if component['is_pass']:
                        st.success(f"🟢 **PASS (符合Derating標準)**")
                    else:
                        st.error(f"❌ **FAIL (未通過Derating標準)**")
                    st.write("---")
                    
            except Exception as e:
                st.error(f"計算失敗，請檢查輸入格式。")
                st.code(f"錯誤報告: {e}")