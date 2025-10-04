import gradio as gr
import pandas as pd
import joblib
import numpy as np
import tempfile

# 1) Load model cuối cùng
pipe = joblib.load("math_score_best_pipe.pkl")

# 2) Hàm dự đoán 1 học sinh
def predict_single(gender, race, parent_edu, lunch, prep, reading, writing):
    row = {
        "gender": gender,
        "race/ethnicity": race,
        "parental level of education": parent_edu,
        "lunch": lunch,
        "test preparation course": prep,
        "reading score": reading,
        "writing score": writing,
    }
    df = pd.DataFrame([row])
    y_pred = pipe.predict(df)[0]
    return float(np.round(y_pred, 2))

# 3) Hàm dự đoán theo CSV
# CSV phải có các cột đầu vào giống lúc train
# Có thể có hoặc không có cột "math score"
def predict_csv(file):
    df = pd.read_csv(file.name)
    X = df.drop(columns=["math score"], errors="ignore")
    preds = pipe.predict(X)
    out = df.copy()
    out["math score predicted"] = np.round(preds, 2)

    # cho phép tải kết quả về
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    out.to_csv(tmp.name, index=False, encoding="utf-8")
    return out, tmp.name

with gr.Blocks(title="Math Score Predictor") as demo:
    gr.Markdown(
        """
        ## Dự đoán Math score
        Nhập thông tin học sinh hoặc tải CSV theo đúng tên cột. 
        Mô hình dùng pipeline đã huấn luyện nên không cần tự xử lý mã hóa.
        """
    )

    with gr.Tab("Dự đoán 1 học sinh"):
        with gr.Row():
            gender = gr.Dropdown(["female", "male"], label="gender", value="female")
            race = gr.Dropdown(
                ["group A", "group B", "group C", "group D", "group E"],
                label="race/ethnicity",
                value="group B",
            )
            parent = gr.Dropdown(
                [
                    "some high school",
                    "high school",
                    "some college",
                    "associate's degree",
                    "bachelor's degree",
                    "master's degree",
                ],
                label="parental level of education",
                value="some college",
            )
        with gr.Row():
            lunch = gr.Dropdown(["standard", "free/reduced"], label="lunch", value="standard")
            prep = gr.Dropdown(["none", "completed"], label="test preparation course", value="none")
        with gr.Row():
            reading = gr.Slider(0, 100, value=70, step=1, label="reading score")
            writing = gr.Slider(0, 100, value=70, step=1, label="writing score")

        btn = gr.Button("Dự đoán")
        out_single = gr.Number(label="Math score dự đoán")

        btn.click(
            predict_single,
            inputs=[gender, race, parent, lunch, prep, reading, writing],
            outputs=out_single,
        )

    with gr.Tab("Dự đoán theo CSV"):
        gr.Markdown(
            """
            CSV cần các cột:
            gender, race/ethnicity, parental level of education, lunch, test preparation course, reading score, writing score.
            Có thể kèm thêm math score để so sánh sau khi dự đoán.
            """
        )
        file_in = gr.File(label="Tải CSV dữ liệu vào")
        btn_csv = gr.Button("Chạy dự đoán CSV")
        out_df = gr.Dataframe(label="Kết quả")
        out_file = gr.File(label="Tải file kết quả")

        btn_csv.click(predict_csv, inputs=file_in, outputs=[out_df, out_file])

if __name__ == "__main__":
    demo.launch(share=True)   # share=True nếu muốn có link public, bỏ nếu chỉ local

