
# Disable Streamlit welcome prompt
import os
os.environ['STREAMLIT_GLOBAL_EMAIL'] = 'no'

# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
try:
    import shap
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
    import shap
import matplotlib.pyplot as plt
import sys
import base64
from io import BytesIO
from xgboost import XGBClassifier

# Configure SHAP to use JS instead of Matplotlib
# shap.initjs()
if st.secrets.get("DEPLOY_ENV") == "cloud":
    pass  # Cloud环境中避免初始化
else:
    shap.initjs()  # 本地环境保留
    
# Configure page
st.set_page_config(
    page_title="Urinary Continence Predictor",
    layout="centered"
)

# Fixed model path
MODEL_PATH = "xgboost_model.pkl"  # 确保模型与代码同目录
# SHAP初始化配置（兼容部署环境）
# shap.initjs()
# shap.js.init()

# Load model
def load_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"MODEL LOADING FAILED: {str(e)}")
        return None

# Input processing function with correct feature order
def process_input(data):
    return pd.DataFrame({
        'BMI': [1 if data['BMI'] >= 24 else 0],
        'MUL': [data['MUL']],
        'LAT': [data['LAT']],
        'LAM_RAD_SCORE': [data['LAM_RAD_SCORE']],
        '手术技术': [1 if data['Nerve_sparing'] == 'Yes (1)' else 0]  # 修改匹配值
    })

# 特征名映射（中文->英文）
FEATURE_MAPPING = {
    'BMI': 'BMI',
    'MUL': 'MUL(mm)',
    'LAT': 'LAT(mm)',
    'LAM_RAD_SCORE': 'LAM RAD Score',
    '手术技术': 'Nerve Sparing'
}

def create_shap_plot(model, df_input):
    """使用JavaScript渲染SHAP图，完美还原本地效果"""
    try:
        # 确保使用TreeExplainer
        if not hasattr(model, 'feature_names_in_'):
            model.feature_names_in_ = df_input.columns.tolist()
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_input)
        
        # 创建英文特征名列表用于显示
        english_features = [FEATURE_MAPPING.get(f, f) for f in df_input.columns]
        
        # 生成SHAP力图
        plot = shap.force_plot(
            base_value=explainer.expected_value,
            shap_values=shap_values[0],
            features=df_input.iloc[0],
            feature_names=english_features,
            matplotlib=False
        )
        
        # 返回完整的HTML并添加自定义样式
        html_content = f"""
        <div style="width:100%; height:300px; overflow:auto;">
            {shap.getjs()}
            {plot.html()}
        </div>
        <style>
            .shap-force-plot {{
                width: 100% !important;
                height: 250px !important;
                font-size: 14px !important;
            }}
            .shap-value {{
                font-size: 14px !important;
            }}
        </style>
        """
        return html_content
    
    except Exception as e:
        return f"<p style='color:red'>SHAP生成错误: {str(e)}</p>"

# Main application
def main():
    # 单一标题 (字体缩小至28px)
    st.markdown(
        """
        <div style="text-align:center; margin-bottom:20px;">
            <h1 style="font-size:28px; color:#2a5298;">
                Prediction Model for Early Urinary Continence Recovery after RARP
            </h1>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # 加载模型
    model = load_model()
    if model is None:
        return

    # 输入参数 - 单列布局
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            lam_score = st.slider('LAM_RAD_SCORE', 
                             min_value=-5.0, max_value=5.0, value=0.01, step=0.01)
            
            mul = st.slider('MUL (mm)', 
                       min_value=5.0, max_value=30.0, value=10.0, step=0.1)
        
        with col2:
            lat = st.slider('LAT (mm)', 
                       min_value=5.0, max_value=30.0, value=12.0, step=0.1)
            
            bmi = st.slider('BMI (1:≥24kg/m2，0:＜24kg/m2)', 
                       min_value=18.0, max_value=35.0, value=25.0, step=0.1)
            st.caption(f"BMI category: {'≥24kg/m2' if bmi >= 24 else '＜24kg/m2'}")
        
        # 神经保留技术单独一行
        nerve = st.radio('Nerve sparing technique', 
                    ('Yes (1)', 'No (0)'), index=0, horizontal=True)
    
    predict_btn = st.button('PREDICT RECOVERY PROBABILITY', type="primary", use_container_width=True)

    # 预测逻辑
    if predict_btn:
        input_data = {
            'LAM_RAD_SCORE': lam_score,
            'MUL': mul,
            'LAT': lat,
            'BMI': bmi,
            'Nerve_sparing': nerve
        }
        
        try:
            df_input = process_input(input_data)
            
            # 确保特征顺序与模型一致
            if hasattr(model, 'feature_names_in_'):
                df_input = df_input[model.feature_names_in_]
            elif hasattr(model, 'get_booster'):
                df_input = df_input[model.get_booster().feature_names]
            
            proba = model.predict_proba(df_input)[0][1]
            prediction = model.predict(df_input)[0]
            
            # 结果展示 - 结果和概率合并为一行
            st.divider()
            if prediction == 1:
                result_text = "✅ CONTINENCE RECOVERED"
                color = "green"
            else:
                result_text = "❗ CONTINENCE NOT RECOVERED"
                color = "red"
            
            # 在同一行显示结果和概率（缩小概率字体）
            st.markdown(
                f"<div style='text-align:center;'>"
                f"<h2 style='display:inline; color:{color};'>{result_text}</h2>"
                f"<span style='font-size:20px; color:{color}; margin-left:15px;'>{proba*100:.1f}% probability</span>"
                f"</div>",
                unsafe_allow_html=True
            )
            
            # 直接显示SHAP图（不显示标题）
            shap_html = create_shap_plot(model, df_input)
            if shap_html:
                st.components.v1.html(shap_html, height=300)  # 使用components渲染HTML
            else:
                st.warning("Could not generate SHAP explanation")
            
        except Exception as e:
            st.error(f"PREDICTION ERROR: {str(e)}")
            st.error(f"Provided features: {list(df_input.columns) if 'df_input' in locals() else 'N/A'}")

if __name__ == '__main__':
    main()
