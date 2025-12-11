# ... (前面的代码保持不变) ...

# ==========================================
# 5. 预测与渲染逻辑 (Core Logic)
# ==========================================
def render_prediction(model, input_data, year):
    # 1. 复制数据
    input_data = input_data.copy()

    # 2. 提取核心模型
    if hasattr(model, 'steps'):
        estimator = model.steps[-1][1]
    else:
        estimator = model

    # 3. 特征对齐
    try:
        if hasattr(estimator, 'feature_names_in_'):
            expected_features = estimator.feature_names_in_
        elif hasattr(estimator, 'feature_names_'):
            expected_features = estimator.feature_names_
        else:
            expected_features = None
        
        if expected_features is not None:
            # 补0并重排
            for col in expected_features:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[list(expected_features)]
    except Exception:
        pass

    # 4. 预测概率
    try:
        esrd_prob = model.predict_proba(input_data)[0][1]
        st.write(f"Probability of kidney failure within {year} year: **{esrd_prob:.2%}**")
    except Exception as e:
        st.error(f"Prediction Error ({year} yr): {e}")
        return

    # 5. SHAP 绘图
    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(input_data)
        
        if isinstance(shap_values, list):
            base_value = explainer.expected_value[1]
            shap_values_plot = shap_values[1]
        else:
            base_value = explainer.expected_value
            shap_values_plot = shap_values

        force_plot = shap.force_plot(
            base_value,
            shap_values_plot,
            input_data,
            matplotlib=False
        )
        
        html_buffer = io.StringIO()
        shap.save_html(html_buffer, force_plot)
        html_content = html_buffer.getvalue()
        
        wrapped = f"<div style='width:100%; overflow-x:auto;'>{html_content}</div>"
        components.html(wrapped, height=150, scrolling=True)

    except Exception:
        st.caption("ℹ️ (Details not available for this model type)")

# ↑↑↑↑↑ render_prediction 函数到这里结束 ↑↑↑↑↑
# ----------------------------------------------------
# ↓↓↓↓↓ 下面的代码必须【顶格写】，不要有缩进！ ↓↓↓↓↓

with right_col:
    st.subheader("🤖 Predicted Results")
    
    # 只有点击按钮后才执行预测
    if predict_btn:
        try:
            # 确定使用哪组模型
            current_models = models_12 if is_full_mode else models_9
            
            # 依次显示 1年、3年、5年 的结果
            st.markdown("#### 1-Year Prediction")
            render_prediction(current_models[1], input_data, 1)
            
            st.markdown("---")
            st.markdown("#### 3-Year Prediction")
            render_prediction(current_models[3], input_data, 3)
            
            st.markdown("---")
            st.markdown("#### 5-Year Prediction")
            render_prediction(current_models[5], input_data, 5)
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
