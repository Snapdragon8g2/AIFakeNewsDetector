# 📰✨ Fake News Detector

## 🚀 Overview  
The **Fake News Detector** is an AI-powered 🧠 application that classifies news articles as **real ✅ or fake ❌**. Using **Natural Language Processing (NLP) 🗣️** and a trained **Logistic Regression Model 📊**, this project helps combat misinformation online!  

## 🔥 Key Features  
- **🧠 Smart ML Model**: Trained using a **balanced dataset** of real and fake news articles.  
- **🌐 Web App Interface**: Built with **Streamlit**, allowing easy URL-based news verification.  
- **📝 Text Cleaning & Preprocessing**: Uses NLP to remove noise and improve prediction accuracy.  
- **🎯 Confidence Score**: Provides a **certainty score** to indicate how confident the model is in its classification.  

## 🛠️ How It Works  
1. **📂 Data Collection**: The model learns from a dataset containing both real & fake news.  
2. **🔍 Text Extraction**: Users enter a **URL**, and the app extracts the main article text via `newspaper3k`.  
3. **📑 Preprocessing**: The extracted text is cleaned 🧹 by removing stop words and non-alphabetic content.  
4. **🤖 Prediction**: The processed text is **vectorized** and fed into the trained model for classification.  
5. **✅ Output & Confidence Score**: The app displays whether the article is **real or fake**, with a **confidence percentage 📊**.  

## 📚 Frameworks & Libraries Used  
- 🖥️ **Streamlit** – For the web interface.  
- 🤖 **scikit-learn** – For model training & evaluation.  
- 📊 **pandas** – For data handling.  
- 📝 **nltk** – For NLP tasks.  
- 📰 **newspaper3k** – For web scraping & content extraction.  
- 🚀 **transformers** – For advanced NLP capabilities (if required).  

## 🎯 Why This Matters  
Misinformation spreads **faster than truth** in the digital age 🌍. The **Fake News Detector** helps people verify news sources 🧐, ensuring they consume only authentic and factual information.  

## ⚡ How to Run Locally  
Follow these steps to get started!  

```bash
# 1️⃣ Create a Virtual Environment 🏗️
python -m venv venv

# 2️⃣ Activate the Environment 🔄
# For Windows:
venv\Scripts\activate  
# For Mac/Linux:
source venv/bin/activate  

# 3️⃣ Upgrade pip 🚀
pip install --upgrade pip

# 4️⃣ Install Required Packages 📦
pip install -r requirements.txt

# 5️⃣ Download NLTK Data 🗄️
python NLTKscript.py

# 6️⃣ Install Additional Dependencies ⚙️
pip install lxml[html_clean]

# 7️⃣ Run the Web App 🎭
streamlit run app.py
```

🎉 **Done!** Now open your browser 🌐 and start verifying news articles like a pro! 🕵️‍♂️✅  

---

## 📌 Contributing  
Feel free to fork this repository and submit a pull request 🤝. Any contributions, from bug fixes to feature additions, are welcome!  

## 💬 Contact  
For any queries, reach out via [GitHub Issues]((https://github.com/Snapdragon8g2))
