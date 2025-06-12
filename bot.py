import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from openai import OpenAI

# Set your OpenAI API key here (for testing you can hardcode it, but use env vars in production)
client= OpenAI(api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

st.title(" CSV Data Visualizer")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df)

    st.subheader("Data Summary")
    st.write(df.describe())

    st.subheader("🧠 Ask Questions About Your Data")
    user_question = st.text_input("Type your question about the CSV data")

    if user_question:
        with st.spinner("Thinking..."):

            # Create a summary of the data
            summary = df.describe(include='all').to_string()
            prompt = f"""
You are a data analyst. A user uploaded this CSV dataset.
Here is a summary of the data:
{summary}

The user asked: "{user_question}"

Analyze the data and provide a clear and helpful answer.
            """

            # Call OpenAI API with new syntax
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            answer = response.choices[0].message.content

            st.success("AI's Response:")
            st.write(answer)

    # Column selection
    st.subheader("Select Columns for Visualization")
    columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if not columns:
        st.warning("No numeric columns found in the uploaded file.")
    else:
        x_axis = st.selectbox("X-axis", columns)
        y_axis = st.selectbox("Y-axis", columns, index=1 if len(columns) > 1 else 0)
        plot_type = st.radio("Select Plot Type", ["Line", "Bar", "Scatter"])

        # Plot
        st.subheader(f"{plot_type} Plot of {y_axis} vs {x_axis}")
        fig, ax = plt.subplots()
        if plot_type == "Line":
            sns.lineplot(data=df, x=x_axis, y=y_axis, ax=ax)
        elif plot_type == "Bar":
            sns.barplot(data=df, x=x_axis, y=y_axis, ax=ax)
        elif plot_type == "Scatter":
            sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
        st.pyplot(fig)
else:
    st.info("Please upload a CSV file to begin.")
