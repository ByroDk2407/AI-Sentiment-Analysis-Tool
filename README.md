# **Real Estate Market Analysis Dashboard**

## **Project Overview**
The **Real Estate Market Analysis Dashboard** is an AI-powered web application that provides **data-driven insights** into the **Australian real estate market**. It integrates **sentiment analysis, economic indicators, and machine learning predictions** to help users understand **current and future market conditions**.

This project leverages:
- **LSTM-based time-series forecasting** to predict future property price trends.
- **Sentiment Fusion** combining **real estate news & social media sentiment**.
- **DeepSeek R1 AI chatbot** for real-time market insights.

## **Key Features**
### **Sentiment Analysis**
- **Visualizes sentiment trends over time** using real estate news and social media data.
- **Breakdown of positive, neutral, and negative sentiments**.

### **Market Predictions**
- Uses **LSTM-based forecasting** to predict **future property prices**.
- Displays **predicted market confidence scores**.

### **Recent Market News**
- Scrapes and analyzes **real estate articles**.
- Extracts **sentiment scores for each article**.

### **Interactive Heatmap (Coming Soon)**
- Displays **real estate sentiment by city/suburb**.

### **AI Chatbot (DeepSeek R1)**
- **Conversational assistant** for real estate insights.
- Uses **retrieval-augmented generation (RAG)** to answer **market trend queries**.

## **Installation & Setup**
### **1️⃣ Install Dependencies**
Ensure you have Python installed as well as postgreSQL, then run:
```bash
pip install -r requirements.txt
```

### **2️⃣ Start the Ollama AI Server**
Pull and run **DeepSeek R1**:
```bash
ollama pull deepseek-chat:1.5b
ollama serve
```

### **3️⃣ Run the Dashboard**
```bash
python app.py
```
The dashboard will be available at `http://localhost:5000`.

## **📜 License**
This project is open-source and licensed under the MIT License. Also not this project is not affiliated with the Australian government or any other government agency. Nor does it provide any financial advice. It is a personal project and should not be used as a source of financial advice.

---
💡 **Contributions & feedback are welcome!**

