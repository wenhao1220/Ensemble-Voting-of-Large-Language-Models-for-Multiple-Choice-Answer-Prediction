import streamlit as st
import pandas as pd
import re
import json
import os
import numpy as np
from ollama import chat

# =========================
# 1. 模型清單
# =========================
MODELS_DICT = {
    "Gemma 3 12B": "gemma3:12b",
    "Qwen 3 8B": "qwen3:8b",
    "DeepSeek-R1 8B": "deepseek-r1:8b",
    "Llama 3.1 8B": "llama3.1:8b",
    "Mistral 7B": "mistral:7b",
    "InternLM 3 (8B)": "huihui_ai/internlm3-abliterated",
    "Phi-4 Mini": "ingu627/phi-4-mini-instruct-q4",
    "GLM4 9B": "glm4:9b",
    "MiniCPM 8B": "myaniu/minicpm4-8b:Q4_K_M"
}

# =========================
# 2. Double Error 演算法邏輯
# =========================
def get_best_models_from_dir(results_dir, k=3):
    if not os.path.exists(results_dir):
        return None, f"⚠️ 找不到目錄: {results_dir}", None

    accuracies = {}
    wrong_vectors = {}
    
    # 掃描目錄下的所有 result_*.xlsx
    files = [f for f in os.listdir(results_dir) if f.startswith("result_") and f.endswith(".xlsx")]
    
    if len(files) < k:
        return None, f"⚠️ 目錄內 result 檔案不足 {k} 個 (目前僅找到 {len(files)} 個)", None

    for filename in files:
        path = os.path.join(results_dir, filename)
        try:
            df_res = pd.read_excel(path)
            model_id = filename.replace("result_", "").replace(".xlsx", "")
            
            # 計算正確率與錯誤向量 (1代表錯, 0代表對)
            actual = df_res['Actual'].astype(str).str.strip()
            pred = df_res['Predicted'].fillna(0).astype(int).astype(str).str.strip()
            is_wrong = (actual != pred).astype(int).values
            
            accuracies[model_id] = 1 - is_wrong.mean()
            wrong_vectors[model_id] = is_wrong
        except:
            continue

    # 挑選正確率前 7 名進行 Double Error 優化 (避免組合過多)
    top_candidates = sorted(accuracies, key=accuracies.get, reverse=True)[:7]
    num_q = len(next(iter(wrong_vectors.values())))
    de_matrix = pd.DataFrame(index=top_candidates, columns=top_candidates)

    for m1 in top_candidates:
        for m2 in top_candidates:
            if m1 == m2: de_matrix.loc[m1, m2] = np.nan
            else:
                # 計算兩個模型同時答錯的機率 (Double Error)
                both_wrong = np.sum((wrong_vectors[m1] == 1) & (wrong_vectors[m2] == 1))
                de_matrix.loc[m1, m2] = both_wrong / num_q

    # 貪婪演算法選出最互補組合
    selected = [top_candidates[0]]
    remaining = [m for m in top_candidates if m not in selected]

    while len(selected) < k and remaining:
        scores = {cand: np.mean([de_matrix.loc[cand, s] for s in selected]) for cand in remaining}
        best_cand = min(scores, key=scores.get)
        selected.append(best_cand)
        remaining.remove(best_cand)
    
    # 對應回 UI 顯示名稱
    final_names = []
    for s in selected:
        found = False
        for ui_name, m_val in MODELS_DICT.items():
            clean_val = m_val.replace(':', '_').replace('/', '_')
            if clean_val == s:
                final_names.append(ui_name)
                found = True
                break
        if not found: final_names.append(s)

    return final_names, None, de_matrix

# =========================
# 3. 工具函式
# =========================
def parse_res(raw_text):
    text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    try:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m: return json.loads(m.group(0))
    except: pass
    return {"answer": None, "confidence": 0, "analysis": {}}

def build_prompt(article, question, opts):
    opt_str = "\n".join([f"{k}. {v}" for k, v in opts.items()])
    return f"""請閱讀文章並回答問題。針對每個選項，請用「繁體中文」詳細說明為何選擇或不選擇的原因。
文章：{article}
問題：{question}
選項：\n{opt_str}

請嚴格輸出 JSON 格式：
{{"answer": 數字, "confidence": 0-1, "analysis": {{"1": "原因", "2": "原因", "3": "原因", "4": "原因"}}}}"""

def query_ollama(prompt, model, think=False):
    try:
        resp = chat(model=model, messages=[{"role": "user", "content": prompt}], 
                    options={"temperature": 0.0}, think=think)
        return resp.message.content
    except Exception as e: return f"Error: {e}"

# =========================
# 4. Streamlit UI
# =========================
st.set_page_config(page_title="AI Ensemble Tool", layout="wide")
st.title("🧪 閱讀理解：模型預測與 Double Error 分析")

if 'q_idx' not in st.session_state: st.session_state.q_idx = 0

with st.sidebar:
    st.header("📂 數據設定")
    uploaded_file = st.file_uploader("1. 上傳測評資料 (c3_m_test.xlsx)", type=["xlsx"])
    results_dir = st.text_input("2. 輸入結果檔目錄 (存放 result_*.xlsx)", value="./results")
    
    st.header("⚙️ 模式設定")
    mode = st.radio("預測模式", ["單一模型", "集成學習"])
    
    if mode == "集成學習":
        method = st.selectbox("演算法", ["Majority Voting (MV)", "Weighted Voting (CWV)", "CCRE"])
        
        # Double Error 自動選擇
        st.subheader("🤖 模型自動挑選")
        if st.button("🔍 執行 Double Error 運算"):
            best, err, matrix = get_best_models_from_dir(results_dir, k=3)
            if err: st.error(err)
            else:
                st.session_state.best_models = best
                st.success(f"最佳組合：{', '.join(best)}")
                with st.expander("查看 Double Error 矩陣"):
                    st.dataframe(matrix.style.background_gradient(cmap='Reds'))
        
        selected_models = st.multiselect("最終參與模型", list(MODELS_DICT.keys()), 
                                         default=st.session_state.get('best_models', list(MODELS_DICT.keys())[:3]))
        think_m = st.selectbox("Thinking Model (CCRE用)", ["deepseek-r1:8b", "qwen3:8b"])
    else:
        one_m = st.selectbox("選擇模型", list(MODELS_DICT.keys()))

# 主畫面
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    
    # 題號控制
    col_a, col_b = st.columns([1, 4])
    with col_a:
        if st.button("🎲 隨機選題"):
            st.session_state.q_idx = np.random.randint(0, len(df))
            st.rerun()
    with col_b:
        st.session_state.q_idx = st.number_input("輸入題號索引", 0, len(df)-1, value=st.session_state.q_idx)

    row = df.iloc[st.session_state.q_idx]
    
    # --- 題目與選項區域 ---
    st.divider()
    st.subheader("📖 文章內容")
    st.info(row['Article'])
    
    st.subheader(f"❓ 問題：{row['Question']}")
    
    # 整理選項
    opts = {i: row[f'Option{i}'] for i in range(1, 5) if pd.notna(row.get(f'Option{i}'))}
    opt_cols = st.columns(len(opts))
    for i, (k, v) in enumerate(opts.items()):
        opt_cols[i].markdown(f"**({k}) {v}**")
    
    # --- 修正：在這裡直接顯示標準答案 ---
    actual = int(row['Answer']) if pd.notna(row['Answer']) else "無"

    # 使用 HTML 確保顏色正確輸出
    st.markdown(f'### 🎯 標準答案：<span style="color:red; font-size:36px; font-weight:bold;">{actual}</span>', unsafe_allow_html=True)
    st.divider()

    # --- 預測按鈕 ---
    if st.button("🚀 開始執行 AI 預測", type="primary"):
        with st.spinner("AI 模型運算中..."):
            prompt = build_prompt(row['Article'], row['Question'], opts)
            
            if mode == "單一模型":
                raw = query_ollama(prompt, MODELS_DICT[one_m])
                res = parse_res(raw)
                final_ans, analysis_dict, src = res.get('answer'), res.get('analysis'), one_m
            else:
                # 集成運算
                results = {m: parse_res(query_ollama(prompt, MODELS_DICT[m])) for m in selected_models}
                preds = {m: r.get('answer') for m, r in results.items()}
                confs = {m: r.get('confidence', 0.5) for m, r in results.items()}
                
                if method == "Majority Voting (MV)":
                    valid_list = [v for v in preds.values() if v is not None]
                    final_ans = max(set(valid_list), key=valid_list.count) if valid_list else None
                    explainer = max([m for m in selected_models if preds[m] == final_ans], key=lambda x: confs[x], default=selected_models[0])
                    analysis_dict, src = results[explainer].get('analysis'), f"MV ({explainer})"
                elif method == "Weighted Voting (CWV)":
                    scores = {i: sum(confs[m] for m in selected_models if preds[m] == i) for i in range(1, 5)}
                    final_ans = max(scores, key=scores.get)
                    explainer = max([m for m in selected_models if preds[m] == final_ans], key=lambda x: confs[x], default=selected_models[0])
                    analysis_dict, src = results[explainer].get('analysis'), f"CWV ({explainer})"
                elif method == "CCRE":
                    m1, m2 = selected_models[0], selected_models[1]
                    if preds[m1] == preds[m2] and preds[m1] is not None:
                        final_ans, analysis_dict, src = preds[m1], results[m1].get('analysis'), f"CCRE ({m1})"
                    else:
                        st.warning("觸發仲裁模型...")
                        r_t = parse_res(query_ollama(prompt, think_m, think=True))
                        final_ans, analysis_dict, src = r_t.get('answer'), r_t.get('analysis'), f"仲裁 ({think_m})"

            # --- 顯示預測結果 ---
            res_c1, res_c2 = st.columns(2)
            res_c1.metric("AI 預測答案", final_ans)
            res_c2.metric("解釋來源模型", src)
            
            if final_ans == actual: st.success("🎉 AI 預測正確！")
            else: st.error("❌ AI 預測錯誤。")
            
            st.subheader("📝 選項詳細解析 (中文)")
            if analysis_dict:
                st.table(pd.DataFrame([{"選項": f"({k})", "分析原因": v} for k, v in analysis_dict.items()]))
            else:
                st.warning("模型未回傳結構化解析內容。")
else:
    st.info("請在側邊欄上傳測評檔案以開始。")