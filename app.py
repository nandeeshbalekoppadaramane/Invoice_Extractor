import streamlit as st
import os
import google.generativeai as genai
import pdfplumber
import pandas as pd
import json
import re
from datetime import datetime
from io import BytesIO

# --- Helper functions ---

def configure_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"❌ Gemini API configuration failed: {e}")
        st.info("Please ensure your API key is correct and valid. You can get one from https://aistudio.google.com/app/apikey")
        return False

def extract_text_from_pdf(uploaded_file):
    page_texts = []
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    page_texts.append(page_text)
                else:
                    st.warning(f"⚠️ No text found on page {i+1} of {uploaded_file.name}. Skipping this page.")
        return page_texts
    except Exception as e:
        st.error(f"❌ Error extracting text from {uploaded_file.name}: {e}")
        st.info("Ensure the PDF is not corrupted or password-protected.")
        return []

def get_gemini_model():
    try:
        # First, try to get gemini-2.5-pro
        try:
            model = genai.GenerativeModel("models/gemini-2.5-flash")
            st.info("✅ Using model: gemini-2.5-pro")
            return model
        except:
            st.warning("⚠️ gemini-2.5-pro not available, trying gemini-1.5-pro...")
        
        # Fallback to gemini-1.5-pro
        try:
            model = genai.GenerativeModel("models/gemini-1.5-pro")
            st.info("✅ Using model: gemini-1.5-pro")
            return model
        except:
            st.warning("⚠️ gemini-1.5-pro not available, trying gemini-1.5-flash...")
        
        # Fallback to gemini-1.5-flash
        for m in genai.list_models():
            if m.name == "models/gemini-1.5-flash" and 'generateContent' in m.supported_generation_methods:
                st.info(f"✅ Using fallback model: {m.name}")
                return genai.GenerativeModel(m.name)
        
        # Last resort: any compatible model
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                st.info(f"✅ Using available model: {m.name}")
                return genai.GenerativeModel(m.name)
        
        st.error("❌ No compatible Gemini model found that supports 'generateContent'.")
        st.info("Please check your API key and available models in Google AI Studio. Your quota might also be exhausted.")
        return None
    except Exception as e:
        st.error(f"❌ Error accessing Gemini models: {e}")
        st.info("Ensure you have a stable internet connection and a valid API key.")
        return None

def parse_invoice_with_gemini(model, invoice_text, file_name="", page_number=None):
    invoice_id = f" (Page {page_number})" if page_number is not None else ""
    
    if not invoice_text.strip():
        st.warning(f"⚠️ Invoice text for {file_name}{invoice_id} is empty, skipping Gemini processing.")
        return None

    prompt = """
You are an intelligent invoice parser for GSTR-1 filing purposes. Your task is to extract information from the given invoice text and determine if it's a B2B or B2C invoice.

**Invoice Type Detection:**
- B2B Invoice: Has a Buyer/Recipient GSTIN (15-character alphanumeric code)
- B2C Invoice: Does NOT have a Buyer/Recipient GSTIN (buyer is an unregistered person/consumer)

**Extract the following information:**

**Common Fields (for both B2B and B2C):**
- Invoice Type: "B2B" or "B2C" (based on presence of Buyer GSTIN)
- Invoice Number
- Invoice Date (format as YYYY-MM-DD if possible)
- Seller GSTIN (15-character GSTIN of the seller/supplier)
- Place of Supply (State name or state code)
- HSN Code (Harmonized System of Nomenclature code)
- Taxable Value (total taxable amount before taxes)
- Invoice Value (total invoice amount including all taxes)
- CGST (Central Goods and Services Tax amount)
- SGST (State Goods and Services Tax amount)
- IGST (Integrated Goods and Services Tax amount)
- Tax Rate (GST rate in percentage, e.g., 18, 12, 5, etc.)

**B2B Specific Fields (extract only if B2B):**
- Party Name (Buyer/Recipient's full company or individual name)
- Buyer GSTIN (15-character GSTIN of the Buyer/Recipient)

**B2C Specific Fields (extract only if B2C):**
- Invoice Type Category: Determine if it's "B2C-Large" (invoice value > 2.5 lakhs) or "B2C-Small" (invoice value ≤ 2.5 lakhs)

**Important Instructions:**
1. Return the result as a JSON object with these exact field names
2. For B2B invoices: Include Party Name and Buyer GSTIN
3. For B2C invoices: Set Party Name to "Consumer" and Buyer GSTIN to "N/A", and include Invoice Type Category
4. If a field is not found, use "N/A" for string values or "0" for numeric values
5. Do not include any extra explanation, markdown, or other text outside the JSON object

Here is the invoice text:
"""
    full_prompt = prompt + invoice_text

    response = None 
    try:
        response = model.generate_content(full_prompt)
        raw_json_str = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw_json_str)

        # Ensure Invoice Type is present
        if "Invoice Type" not in data:
            # Try to determine based on Buyer GSTIN
            buyer_gstin = data.get("Buyer GSTIN", "N/A")
            if buyer_gstin != "N/A" and len(buyer_gstin) == 15:
                data["Invoice Type"] = "B2B"
            else:
                data["Invoice Type"] = "B2C"

        # Set defaults based on invoice type
        if data.get("Invoice Type") == "B2C":
            data["Party Name"] = "Consumer"
            data["Buyer GSTIN"] = "N/A"
            
            # Determine B2C category if not already set
            if "Invoice Type Category" not in data:
                try:
                    invoice_value = float(str(data.get("Invoice Value", 0)).replace(',', ''))
                    data["Invoice Type Category"] = "B2C-Large" if invoice_value > 250000 else "B2C-Small"
                except:
                    data["Invoice Type Category"] = "N/A"
        else:
            # For B2B, remove B2C specific field if present
            data.pop("Invoice Type Category", None)

        # Clean numeric fields
        numeric_fields = ["Taxable Value", "Invoice Value", "CGST", "SGST", "IGST", "Tax Rate"]
        for field in numeric_fields:
            if field in data and isinstance(data[field], str):
                try:
                    clean_value = data[field].replace('$', '').replace('€', '').replace('£', '').replace('₹', '').replace(',', '').replace('%', '').strip()
                    data[field] = float(clean_value) if clean_value else 0.0
                except ValueError:
                    data[field] = 0.0

        # Handle missing HSN Code
        if "HSN Code" not in data:
            data["HSN Code"] = "Not found"

        # Handle missing Place of Supply
        if "Place of Supply" not in data:
            data["Place of Supply"] = "N/A"

        return data
    except json.JSONDecodeError as je:
        st.error(f"❌ Failed to parse Gemini's JSON response for {file_name}{invoice_id}: {je}")
        st.code(f"Raw Response from Gemini (could not parse):\n{response.text}" if response else "No response received.")
        return None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during Gemini processing for {file_name}{invoice_id}: {e}")
        st.code(f"Raw Response from Gemini (if available):\n{response.text}" if response else "No response received.")
        return None

# --- Streamlit App Layout ---
st.set_page_config(page_title="Gemini Invoice Parser - B2B & B2C", layout="centered")

st.title("📄 Invoice Extraction Tool - B2B & B2C (GSTR-1)")

st.markdown("""
Welcome to the Invoice Extraction Tool for GSTR-1 Filing!

This tool automatically:
- ✅ Identifies whether an invoice is **B2B** or **B2C**
- ✅ Extracts relevant fields for GSTR-1 filing
- ✅ Handles both single and merged PDF invoices
- ✅ Categorizes B2C invoices as Large (>2.5L) or Small (≤2.5L)

Upload your PDF invoices and let Gemini AI do the work!
""")

api_key = st.text_input("🔑 Enter your Gemini API Key:", type="password", help="Your API key will not be stored.")
uploaded_files = st.file_uploader("📁 Upload PDF Invoice(s):", type="pdf", accept_multiple_files=True)

if st.button("🚀 Process Invoices"):
    if not api_key:
        st.error("❌ Please enter your Gemini API Key to proceed.")
        st.stop()
    elif not uploaded_files:
        st.warning("⚠️ Please upload at least one PDF file to process.")
        st.stop()
    else:
        st.info("Starting invoice processing... This might take a moment depending on file size and number of invoices.")

        with st.spinner("Configuring Gemini API..."):
            if not configure_gemini(api_key):
                st.stop()

        with st.spinner("Finding a suitable Gemini model..."):
            model = get_gemini_model()
            if not model:
                st.stop()

        all_extracted_invoices_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_files_to_process = len(uploaded_files)
        current_file_index = 0

        for uploaded_file in uploaded_files:
            current_file_index += 1
            status_text.text(f"Processing file {current_file_index}/{total_files_to_process}: {uploaded_file.name}")
            uploaded_file.seek(0)

            with st.spinner(f"Extracting text from {uploaded_file.name}..."):
                page_texts = extract_text_from_pdf(uploaded_file)
                if not page_texts:
                    st.error(f"Failed to extract any text pages from {uploaded_file.name}. Skipping.")
                    progress_bar.progress(current_file_index / total_files_to_process)
                    continue
            
            st.info(f"Detected {len(page_texts)} page(s) in {uploaded_file.name}. Processing each page as a potential invoice.")

            for page_idx, page_text_content in enumerate(page_texts):
                invoice_identifier = f"{uploaded_file.name} (Page {page_idx+1})"
                
                with st.spinner(f"Sending page {page_idx+1} from {uploaded_file.name} to Gemini..."):
                    parsed_data = parse_invoice_with_gemini(model, page_text_content, invoice_identifier, page_idx + 1)
                    
                if parsed_data:
                    parsed_data['Source_File'] = uploaded_file.name
                    parsed_data['Original_File_Page'] = page_idx + 1
                    all_extracted_invoices_data.append(parsed_data)
                    
                    invoice_type = parsed_data.get('Invoice Type', 'Unknown')
                    if invoice_type == "B2C":
                        category = parsed_data.get('Invoice Type Category', 'N/A')
                        st.success(f"✅ Data extracted from {invoice_identifier} - Type: {invoice_type} ({category})")
                    else:
                        st.success(f"✅ Data extracted from {invoice_identifier} - Type: {invoice_type}")
                else:
                    st.error(f"❌ Failed to extract structured data from {invoice_identifier}. See errors above.")
            
            progress_bar.progress(current_file_index / total_files_to_process)

        status_text.text("Invoice processing complete!")
        progress_bar.empty()

        if all_extracted_invoices_data:
            st.success("🎉 All invoices processed! Your Excel file is ready.")
            
            # Display summary statistics
            df_temp = pd.DataFrame(all_extracted_invoices_data)
            b2b_count = len(df_temp[df_temp.get('Invoice Type') == 'B2B'])
            b2c_count = len(df_temp[df_temp.get('Invoice Type') == 'B2C'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Invoices", len(all_extracted_invoices_data))
            with col2:
                st.metric("B2B Invoices", b2b_count)
            with col3:
                st.metric("B2C Invoices", b2c_count)
            
            try:
                df = pd.DataFrame(all_extracted_invoices_data)
                
                # Reorder columns for better readability
                primary_cols = ['Source_File', 'Original_File_Page', 'Invoice Type']
                remaining_cols = [col for col in df.columns if col not in primary_cols]
                df = df[primary_cols + remaining_cols]

                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Write all invoices to one sheet
                    df.to_excel(writer, index=False, sheet_name='All Invoices')
                    
                    # Separate B2B and B2C invoices into different sheets
                    if b2b_count > 0:
                        df_b2b = df[df['Invoice Type'] == 'B2B'].copy()
                        df_b2b.to_excel(writer, index=False, sheet_name='B2B Invoices')
                    
                    if b2c_count > 0:
                        df_b2c = df[df['Invoice Type'] == 'B2C'].copy()
                        df_b2c.to_excel(writer, index=False, sheet_name='B2C Invoices')
                
                processed_data = output.getvalue()

                st.download_button(
                    label="📥 Download Extracted Data as Excel",
                    data=processed_data,
                    file_name=f"extracted_invoices_B2B_B2C_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                
                # Display preview
                st.subheader("📊 Data Preview")
                st.dataframe(df.head(10))
                
            except Exception as e:
                st.error(f"❌ Error preparing or saving data to Excel: {e}")
                st.info("Please ensure 'openpyxl' is installed (pip install openpyxl) and try again.")
        else:
            st.warning("No data could be successfully extracted from any of the uploaded PDFs.")

st.markdown("---")
st.markdown("""
**Note:** 
- B2B invoices require Buyer GSTIN (15-digit number)
- B2C invoices are for consumers without GSTIN
- B2C-Large: Invoice value > ₹2,50,000
- B2C-Small: Invoice value ≤ ₹2,50,000
""")
st.write("Developed using Streamlit and Gemini AI.")
