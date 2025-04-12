# 🧾 BILLBOT

**BILLBOT** is a Streamlit-based web app that digitizes physical bills by extracting raw text using **OCR**, organizing the content using **LLMs (MistralAI)**, and appending structured data to a user-provided **Excel file**.

Say goodbye to manual data entry — BILLBOT makes bill processing clean, fast, and intelligent.

---

## 🚀 Features

- 📸 **OCR with MistralAI**: Converts scanned or photographed bills into raw text.
- 🧠 **Field Extraction with LLM (MistralAI)**: Automatically identifies key fields like:
  - Vendor name
  - Date
  - Items
  - Tax
  - Total amount
- 🖥️ **Streamlit Web Interface**: Simple, user-friendly frontend.
- 📊 **Excel Integration**: Appends structured bill data to an existing Excel sheet.
- 📁 **Supports Multiple File Uploads**

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **OCR & LLM:** MistralAI  
- **Excel Handling:** `pandas`, `openpyxl`  
- **Backend Language:** Python

---

