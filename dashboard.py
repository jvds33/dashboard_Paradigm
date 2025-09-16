import streamlit as st
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys
import zipfile
import io
import shutil

# Import the plotting functions from kto_cards.py
from kto_cards import run_all

def create_download_zip(plot_paths, label_name):
    """Create a ZIP file containing all plots in a folder named after the label."""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for plot_key, plot_path in plot_paths.items():
            if plot_path and os.path.exists(plot_path):
                # Create the file name with label folder structure
                file_name = f"{label_name}/{plot_key}.png"
                zip_file.write(plot_path, file_name)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# Set page config
st.set_page_config(
    page_title="KTO Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stButton > button {
        background-color: #10B981;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 0.5rem;
        font-weight: bold;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #059669;
    }
    .stDownloadButton > button {
        background-color: #7C3AED;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        width: 100%;
    }
    .stDownloadButton > button:hover {
        background-color: #6D28D9;
    }
    .plot-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📊 KTO Dashboard Generator")
st.markdown("Upload your Excel file, select a sheet and label, and generate comprehensive KTO visualizations.")

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=['xlsx', 'xls'],
        help="Upload your KTO Excel file containing survey data"
    )
    
    # Initialize session state
    if 'labels' not in st.session_state:
        st.session_state.labels = []
    if 'selected_label' not in st.session_state:
        st.session_state.selected_label = None
    if 'plots_generated' not in st.session_state:
        st.session_state.plots_generated = False
    if 'plot_paths' not in st.session_state:
        st.session_state.plot_paths = {}
    if 'sheets' not in st.session_state:
        st.session_state.sheets = []
    if 'selected_sheet' not in st.session_state:
        st.session_state.selected_sheet = None
    if 'customizations' not in st.session_state:
        st.session_state.customizations = {}
    if 'df_columns' not in st.session_state:
        st.session_state.df_columns = []
    if 'excel_temp_path' not in st.session_state:
        st.session_state.excel_temp_path = None
    if 'temp_output_dir' not in st.session_state:
        st.session_state.temp_output_dir = None
    if 'edit_open' not in st.session_state:
        st.session_state.edit_open = {}

# Main content
if uploaded_file is not None:
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_excel_path = tmp_file.name
        # Persist the uploaded Excel path for subsequent edits/regenerations
        try:
            prev_path = st.session_state.get('excel_temp_path')
            if prev_path and prev_path != temp_excel_path and os.path.exists(prev_path):
                try:
                    os.unlink(prev_path)
                except Exception:
                    pass
            st.session_state.excel_temp_path = temp_excel_path
        except Exception:
            st.session_state.excel_temp_path = temp_excel_path
        
        # Try to read the Excel file and get sheet names
        try:
            # Get all sheet names
            xl_file = pd.ExcelFile(temp_excel_path)
            st.session_state.sheets = xl_file.sheet_names
            
            with st.sidebar:
                # Sheet selection
                if st.session_state.sheets:
                    st.session_state.selected_sheet = st.selectbox(
                        "Select a sheet",
                        options=st.session_state.sheets,
                        help="Choose the sheet that contains your survey data"
                    )
                    
                    # Only proceed if a sheet is selected
                    if st.session_state.selected_sheet:
                        # Try to read the selected sheet and extract labels
                        try:
                            df = pd.read_excel(temp_excel_path, sheet_name=st.session_state.selected_sheet)
                            
                            # Look for Label column (case-insensitive)
                            label_column = None
                            for col in df.columns:
                                if col.lower() in ['label', 'labels']:
                                    label_column = col
                                    break
                            
                            if label_column is not None:
                                # Get unique labels
                                unique_labels = df[label_column].dropna().unique().tolist()
                                st.session_state.labels = sorted(unique_labels)
                                st.session_state.df_columns = list(df.columns)
                                
                                st.success(f"✅ Sheet '{st.session_state.selected_sheet}' loaded successfully!")
                                st.info(f"Found {len(df)} rows and {len(st.session_state.labels)} unique labels")
                                
                                # Label selection
                                if st.session_state.labels:
                                    st.session_state.selected_label = st.selectbox(
                                        "Select a label",
                                        options=st.session_state.labels,
                                        help="Choose the label to filter data and generate plots for"
                                    )
                                    
                                    # Generate button
                                    if st.button("🚀 Generate Dashboard", type="primary"):
                                        if st.session_state.selected_label:
                                            with st.spinner("Generating plots... This may take a few moments."):
                                                try:
                                                    # Create temporary output directory
                                                    temp_output_dir = tempfile.mkdtemp()
                                                    
                                                    # Run the plotting function
                                                    plot_paths = run_all(
                                                        excel=temp_excel_path,
                                                        sheet=st.session_state.selected_sheet,
                                                        label=st.session_state.selected_label,
                                                        outdir=temp_output_dir,
                                                        customizations=st.session_state.customizations
                                                    )
                                                    
                                                    st.session_state.plot_paths = plot_paths
                                                    st.session_state.plots_generated = True
                                                    st.session_state.excel_temp_path = temp_excel_path
                                                    st.session_state.temp_output_dir = temp_output_dir
                                                    
                                                    st.success("✅ Dashboard generated successfully!")
                                                    
                                                except Exception as e:
                                                    st.error(f"❌ Error generating plots: {str(e)}")
                                                    st.session_state.plots_generated = False
                                        else:
                                            st.warning("Please select a label first!")
                            else:
                                st.error("❌ No 'Label' column found in the selected sheet!")
                                st.error(f"**No Label Column Found**\n\nThe selected sheet '{st.session_state.selected_sheet}' doesn't contain a 'Label' column. Please make sure your sheet has a column named 'Label' with the different categories/labels you want to filter by.")
                                
                        except Exception as e:
                            st.error(f"❌ Error reading sheet '{st.session_state.selected_sheet}': {str(e)}")
                            st.error(f"**Error Reading Sheet**\n\n{str(e)}\n\nPlease make sure the selected sheet is valid and contains the expected data format.")
                    
                else:
                    st.error("❌ No sheets found in the Excel file!")
                    
        except Exception as e:
            st.sidebar.error(f"❌ Error reading Excel file: {str(e)}")
            st.error(f"**Error Reading File**\n\n{str(e)}\n\nPlease make sure your Excel file is valid.")
        
        # Do not delete temp_excel_path here; it's kept in session_state for edits.
            
    except Exception as e:
        st.error(f"❌ Error processing uploaded file: {str(e)}")

else:
    # Welcome message when no file is uploaded
    st.markdown("""
    ## Welcome to the KTO Dashboard Generator! 👋
    
    This dashboard helps you create comprehensive visualizations from your KTO (Customer Satisfaction) survey data.
    
    ### How to use:
    1. **Upload your Excel file** using the file uploader in the sidebar
    2. **Select a sheet** from the dropdown menu (based on available sheets in your Excel file)
    3. **Select a label** from the dropdown menu (based on the Label column in your data)
    4. **Click "Generate Dashboard"** to create all visualizations
    5. **View and download** your generated plots
    
    ### What you'll get:
    - 📊 Respondent count analysis
    - 📈 NPS (Net Promoter Score) visualization
    - 🎯 Service satisfaction breakdowns
    - 📋 Detailed aspect comparisons
    - And much more!
    
    ### Requirements:
    - Excel file (.xlsx or .xls format)
    - Must contain a 'Label' column for filtering
    
    Get started by uploading your Excel file! 🚀
    """)

# Display generated plots
if st.session_state.plots_generated and st.session_state.plot_paths:
    st.markdown("---")
    st.header(f"📊 Dashboard for: {st.session_state.selected_label}")
    
    # Plot titles mapping
    plot_titles = {
        'respondent_count': '👥 Respondent Count Overview',
        'aanbeveling': '📊 Recommendation Distribution',
        'nps_gauge': '🎯 Net Promoter Score (NPS)',
        'duur': '⏱️ Duration of Collaboration',
        'diensten': '🛠️ Services Utilized',
        'info_pie': 'ℹ️ Information Provision',
        'omgang_info': '🔒 Information Handling',
        'smileys': '😊 Impact Assessment',
        'ja_nee': '✅ Collaboration Choice Involvement',
        'aspecten_likert': '📋 Professional Aspects Comparison',
        'aspecten_systeem_likert': '💻 System Aspects Comparison',
        'priority_matrix': '📌 Priority Matrix',
        'open_antwoord_analyse': '📝 Open Answer Text Analysis',
        'functiegroep': '👤 Functiegroep'
    }

    # Default titles used inside the plots (card titles)
    default_plot_titles = {
        'respondent_count': 'Aantal respondenten',
        'aanbeveling': "Aanbeveling bij collega's en relaties",
        'nps_gauge': 'NPS',
        'duur': 'Duur samenwerking',
        'diensten': 'Afgenomen  diensten',
        'info_pie': 'Informatievoorziening',
        'omgang_info': 'Omgang informatie',
        'smileys': 'Impact op verzuim, inzetbaarheid\nen preventie',
        'aspecten_likert': 'Vergelijking van aspecten professionals',
        'aspecten_systeem_likert': 'Vergelijking van aspecten systeem',
        'priority_matrix': 'Prioriteitenmatrix',
        'open_antwoord_analyse': 'Open antwoord tekstanalyse',
        'functiegroep': 'Functiegroep',
        'ja_nee': 'Betrokken bij keuze samenwerking'
    }

    # Default sizes for plots
    default_sizes = {
        'respondent_count': (11.0, 6.0),
        'aanbeveling': (11.5, 6.0),
        'nps_gauge': (7.2, 4.6),
        'duur': (11.0, 4.8),
        'diensten': (12.0, 6.8),
        'info_pie': (12.8, 4.4),
        'omgang_info': (12.8, 4.4),
        'smileys': (12.0, 5.8),
        'aspecten_likert': (16.0, 8.0),
        'aspecten_systeem_likert': (16.0, 8.0),
        'priority_matrix': (16.0, 12.0),
        'open_antwoord_analyse': (16.0, 10.0),
        'functiegroep': (12.0, 6.8)
    }

    # Plots that support a 'column' override
    supports_column = {
        'aanbeveling', 'nps_gauge', 'duur', 'info_pie', 'omgang_info', 'smileys', 'functiegroep'
    }

    # Default columns per plot (used if user hasn't customized yet)
    default_column_by_plot = {
        'aanbeveling': 'Blok1_NPS_KPI_page6_Text',
        'nps_gauge': 'Blok1_NPS_KPI_page6_Text',
        'duur': 'Blok7_3 Duur samenwerking_page74_Text',
        'info_pie': 'Blok7_VoorafGeinformeerd_Resolu_page128_Text',
        'omgang_info': 'Blok8_TevredenheidPrivacyInformatie_page114_Text',
        'smileys': 'Blok1_Alg1_page9_Text',
        'functiegroep': 'Blok4_1_page22_Text'
    }
    
    # Create columns for better layout
    cols = st.columns(2)
    
    for i, (plot_key, plot_path) in enumerate(st.session_state.plot_paths.items()):
        if plot_path and os.path.exists(plot_path):
            with cols[i % 2]:
                st.subheader(plot_titles.get(plot_key, plot_key.replace('_', ' ').title()))
                
                try:
                    st.image(plot_path, use_column_width=True)
                    
                    # Download button for each plot
                    with open(plot_path, 'rb') as file:
                        st.download_button(
                            label=f"📥 Download {plot_titles.get(plot_key, plot_key)}",
                            data=file.read(),
                            file_name=f"{plot_key}_{st.session_state.selected_label}.png",
                            mime="image/png",
                            key=f"download_{plot_key}"
                        )

                    # Edit controls
                    edit_col1, edit_col2 = st.columns([1, 3])
                    with edit_col1:
                        if st.button("✏️ Edit", key=f"edit_btn_{plot_key}"):
                            st.session_state.edit_open[plot_key] = not st.session_state.edit_open.get(plot_key, False)
                    with edit_col2:
                        if st.session_state.edit_open.get(plot_key, False):
                            with st.expander("Edit settings", expanded=True):
                                # Title
                                current_title = st.session_state.customizations.get(plot_key, {}).get('title', default_plot_titles.get(plot_key, ''))
                                new_title = st.text_input("Title", value=current_title, key=f"title_{plot_key}")
                                
                                # Size
                                cur_size = st.session_state.customizations.get(plot_key, {}).get('figsize', default_sizes.get(plot_key, (12.0, 6.0)))
                                width = st.number_input("Width", min_value=4.0, max_value=48.0, step=0.5, value=float(cur_size[0]), key=f"w_{plot_key}")
                                height = st.number_input("Height", min_value=3.0, max_value=48.0, step=0.5, value=float(cur_size[1]), key=f"h_{plot_key}")
                                
                                # Column (where applicable)
                                selected_column = None
                                if plot_key in supports_column and st.session_state.df_columns:
                                    default_col = st.session_state.customizations.get(plot_key, {}).get('column', default_column_by_plot.get(plot_key))
                                    if default_col not in st.session_state.df_columns:
                                        default_col = None
                                    selected_column = st.selectbox(
                                        "Column",
                                        options=st.session_state.df_columns,
                                        index=(st.session_state.df_columns.index(default_col) if default_col in st.session_state.df_columns else 0),
                                        key=f"col_{plot_key}"
                                    )
                                
                                # Save changes
                                if st.button("✅ Apply", key=f"apply_{plot_key}"):
                                    # Update customizations
                                    st.session_state.customizations.setdefault(plot_key, {})
                                    st.session_state.customizations[plot_key]['title'] = new_title
                                    st.session_state.customizations[plot_key]['width'] = float(width)
                                    st.session_state.customizations[plot_key]['height'] = float(height)
                                    if plot_key in supports_column and selected_column:
                                        st.session_state.customizations[plot_key]['column'] = selected_column
                                    
                                    # Regenerate plots with customizations
                                    try:
                                        with st.spinner("Updating plot..."):
                                            updated_paths = run_all(
                                                excel=st.session_state.excel_temp_path,
                                                sheet=st.session_state.selected_sheet,
                                                label=st.session_state.selected_label,
                                                outdir=st.session_state.temp_output_dir or tempfile.mkdtemp(),
                                                customizations=st.session_state.customizations
                                            )
                                            st.session_state.plot_paths = updated_paths
                                            st.success("Plot updated!")
                                    except Exception as e:
                                        st.error(f"Failed to update plot: {str(e)}")
                                    
                                    # Close editor
                                    st.session_state.edit_open[plot_key] = False
                except Exception as e:
                    st.error(f"Error displaying {plot_key}: {str(e)}")
    
    # Summary and download all
    st.markdown("---")
    
    # Create two columns for summary and download all button
    summary_col, download_col = st.columns([3, 1])
    
    with summary_col:
        st.markdown(f"""
        ### 📈 Summary
        Generated **{len(st.session_state.plot_paths)}** visualizations for label: **{st.session_state.selected_label}**
        
        Each plot provides insights into different aspects of your KTO survey data:
        - Customer satisfaction levels
        - Service utilization patterns  
        - Collaboration effectiveness
        - System and professional performance metrics
        """)
    
    with download_col:
        st.markdown("### 📦 Download All")
        
        # Create ZIP file with all plots
        try:
            zip_data = create_download_zip(st.session_state.plot_paths, st.session_state.selected_label)
            
            # Sanitize label name for filename
            safe_label = "".join(c for c in st.session_state.selected_label if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_label = safe_label.replace(' ', '_')
            
            st.download_button(
                label="📥 Download All Plots",
                data=zip_data,
                file_name=f"KTO_Dashboard_{safe_label}.zip",
                mime="application/zip",
                help=f"Download all plots in a ZIP file with folder structure: {st.session_state.selected_label}/",
                type="primary"
            )
            
            st.info(f"📁 **{len(st.session_state.plot_paths)}** plots will be downloaded in folder: `{st.session_state.selected_label}/`")
            
        except Exception as e:
            st.error(f"Error creating download package: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em; margin-top: 2rem;'>
    KTO Dashboard Generator | Built with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)
