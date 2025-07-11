
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
        '手术技术': [1 if data['Nerve_sparing'] == 'Yes' else 0]  # 修正为模型要求的特征名
    })

# 特征名映射（中文->英文）
FEATURE_MAPPING = {
    'BMI': 'BMI(kg/m2)',
    'MUL': 'MUL(mm)',
    'LAT': 'LAT(mm)',
    'LAM_RAD_SCORE': 'LAM RAD Score',
    '手术技术': 'Nerve Sparing'
}

def create_shap_plot(model, df_input):
    """使用自定义格式的JavaScript渲染SHAP图，解决左侧标签截断问题"""
    try:
        if not hasattr(model, 'feature_names_in_'):
            model.feature_names_in_ = df_input.columns.tolist()
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_input)
        base_value = explainer.expected_value
        
        custom_features = []
        for i, feat in enumerate(df_input.columns):
            value = df_input.iloc[0, i]
            if feat == '手术技术':
                display_value = "YES" if value == 1 else "NO"
                custom_features.append(f"{FEATURE_MAPPING[feat]}: {display_value}")
            elif feat == 'BMI':
                display_value = "≥24" if value == 1 else "<24"
                custom_features.append(f"{FEATURE_MAPPING[feat]}: {display_value}")
            else:
                custom_features.append(f"{FEATURE_MAPPING[feat]}: {value:.2f}")
        
        plot = shap.force_plot(
            base_value=base_value,
            shap_values=shap_values[0],
            features=custom_features,
            feature_names=custom_features,
            matplotlib=False,
            show=False
        )
        
        # 关键修改：全新布局方案解决标签截断问题
        html_content = f"""
        <!-- 增加外层容器并添加左侧缓冲空间 -->
        <div style="width:100%; min-width:500px; margin-left:80px; overflow:visible">
            {shap.getjs()}
            <div id="shap-container" style="width:100%; height:280px">
                {plot.html()}
            </div>
        </div>
        
        <style>
            /* 强制重置SHAP默认样式 */
            .shap-force-plot {{
                width: calc(100% + 100px) !important;
                min-width: 500px !important;
                height: 240px !important;
                transform: translateX(40px);
            }}
            
            /* 增加左侧标签显示空间 */
            .shap-left-panel {{
                min-width: 220px !important;
                padding-left: 40px !important;
            }}
            
            /* 保证标签完整可见 */
            .shap-force-plot .feature-value {{
                max-width: none !important;
                overflow: visible !important;
                text-overflow: unset !important;
            }}
            
            /* 调整数值位置 */
            .shap-value {{
                transform: translateX(5px);
                font-size: 11.5px !important;
            }}
            
            /* 基础值标签优化 */
            .shap-base-value {{
                left: -50px !important;
            }}
            
            /* 解决SVG裁剪问题 */
            svg.shap-svg {{
                overflow: visible !important;
            }}
        </style>
        
        <script>
            // 渲染完成后强制刷新布局
            setTimeout(() => {{
                // 格式化数值标签
                document.querySelectorAll('text.shap-value').forEach(el => {{
                    if (!isNaN(el.textContent)) {{
                        el.textContent = parseFloat(el.textContent).toFixed(1);
                    }}
                }});
                
                // 格式化基础值
                const baseEl = document.querySelector('text.shap-base-value');
                if (baseEl) baseEl.textContent = parseFloat(baseEl.textContent).toFixed(2);
                
                // 动态调整标签空间
                const containers = document.querySelectorAll('.shap-left-panel');
                containers.forEach(el => {{
                    el.style.minWidth = '220px';
                    el.style.paddingRight = '10px';
                }});
                
                // 刷新SVG渲染区域
                const svg = document.querySelector('svg.shap-svg');
                if (svg) {{
                    const bbox = svg.getBBox();
                    svg.setAttribute('viewBox', `${{bbox.x-50}} ${{bbox.y}} ${{bbox.width+80}} ${{bbox.height}}`);
                }}
            }}, 800);
        </script>
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
                       min_value=5.0, max_value=20.0, value=10.0, step=0.1)
        
        with col2:
            lat = st.slider('LAT (mm)', 
                       min_value=5.0, max_value=20.0, value=12.0, step=0.1)
            
            bmi = st.slider('BMI', 
                       min_value=18.0, max_value=35.0, value=25.0, step=0.1)
            st.caption(f"BMI category: {'≥24kg/m2' if bmi >= 24 else '＜24kg/m2'}")
        
        # 神经保留技术单独一行
        nerve = st.radio('Nerve sparing technique', 
                    ('Yes', 'No'), index=0, horizontal=True)
    
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
