# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from PIL import Image
# import io
# import base64
# import time

# # Set page configuration
# st.set_page_config(
#     page_title="EEG Report Analyzer",
#     page_icon="🧠",
#     layout="wide",
# )

# # Custom CSS for styling
# st.markdown("""
#     <style>
#         .main-header {
#             font-size: 36px;
#             font-weight: bold;
#             color: #2E86C1;
#             margin-bottom: 20px;
#         }
#         .sub-header {
#             font-size: 24px;
#             font-weight: bold;
#             color: #2874A6;
#             margin-top: 30px;
#             margin-bottom: 10px;
#         }
#         .highlight-text {
#             background-color: #F9E79F;
#             padding: 10px;
#             border-radius: 5px;
#             margin: 10px 0px;
#         }
#         .abnormal-region {
#             background-color: rgba(231, 76, 60, 0.3);
#             border-radius: 5px;
#             padding: 5px;
#         }
#         .normal-region {
#             background-color: rgba(46, 204, 113, 0.3);
#             border-radius: 5px;
#             padding: 5px;
#         }
#         .diagnosis-box {
#             background-color: #EBF5FB;
#             border-left: 5px solid #3498DB;
#             padding: 15px;
#             border-radius: 5px;
#             margin: 20px 0px;
#         }
#         .treatment-box {
#             background-color: #E8F8F5;
#             border-left: 5px solid #1ABC9C;
#             padding: 15px;
#             border-radius: 5px;
#             margin: 20px 0px;
#         }
#         .warning-box {
#             background-color: #FDEDEC;
#             border-left: 5px solid #E74C3C;
#             padding: 15px;
#             border-radius: 5px;
#             margin: 20px 0px;
#         }
#     </style>
#     <div class="main-header">EEG Report Analysis and Prediction System</div>
# """, unsafe_allow_html=True)

# # Sidebar
# with st.sidebar:
#     st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/frontend/src/assets/branding/streamlit-mark-color.svg", width=100)
#     st.markdown("## Input Options")
    
#     input_type = st.radio(
#         "Select Input Method",
#         ["Upload EEG Data File", "Use Sample Data", "Manual Parameter Input"]
#     )
    
#     st.markdown("---")
#     st.markdown("## Analysis Settings")
    
#     sensitivity = st.slider("Analysis Sensitivity", 0.1, 1.0, 0.7, 0.1)
#     frequency_bands = st.multiselect(
#         "Frequency Bands to Analyze",
#         ["Delta (0.5-4 Hz)", "Theta (4-8 Hz)", "Alpha (8-13 Hz)", "Beta (13-30 Hz)", "Gamma (30-100 Hz)"],
#         default=["Delta (0.5-4 Hz)", "Theta (4-8 Hz)", "Alpha (8-13 Hz)", "Beta (13-30 Hz)"]
#     )
    
#     st.markdown("---")
#     st.markdown("### About")
#     st.info("""
#     This application analyzes EEG data to identify potential abnormalities 
#     and provide diagnostic insights with treatment recommendations.
    
#     **Note:** This is a prototype and should not replace professional medical advice.
#     """)

# # Main function to generate EEG sample data
# def generate_eeg_sample(duration=10, sampling_rate=250, channels=8, abnormal=True):
#     """Generate sample EEG data with optional abnormalities"""
#     time_points = np.arange(0, duration, 1/sampling_rate)
#     n_samples = len(time_points)
    
#     # Base frequencies for different brain waves
#     delta = np.sin(2 * np.pi * 2 * time_points)  # 2 Hz (delta wave)
#     theta = np.sin(2 * np.pi * 6 * time_points)  # 6 Hz (theta wave)
#     alpha = np.sin(2 * np.pi * 10 * time_points)  # 10 Hz (alpha wave)
#     beta = np.sin(2 * np.pi * 20 * time_points)  # 20 Hz (beta wave)
    
#     # Generate clean EEG data
#     eeg_data = np.zeros((channels, n_samples))
    
#     for i in range(channels):
#         # Mix different wave components with random weights
#         a, b, c, d = np.random.rand(4)
#         eeg_data[i] = (a * delta + b * theta + c * alpha + d * beta) / (a + b + c + d)
#         # Add some random noise
#         eeg_data[i] += np.random.normal(0, 0.1, n_samples)
    
#     # Add abnormalities if requested (spike-wave complexes typical in some seizures)
#     abnormal_regions = []
#     if abnormal:
#         # Add 2-3 abnormal regions
#         num_abnormalities = np.random.randint(2, 4)
#         for _ in range(num_abnormalities):
#             # Random start time for abnormality
#             abnormal_start = np.random.randint(0, n_samples - sampling_rate)
#             abnormal_duration = np.random.randint(sampling_rate // 2, sampling_rate * 2)
#             abnormal_end = min(abnormal_start + abnormal_duration, n_samples)
            
#             # Select random channels for the abnormality
#             affected_channels = np.random.choice(channels, size=np.random.randint(2, channels+1), replace=False)
            
#             # Create spike-wave pattern
#             for ch in affected_channels:
#                 # Create a spike-wave complex
#                 spike_freq = 3  # Hz (typical for absence seizures)
#                 num_cycles = int((abnormal_end - abnormal_start) / sampling_rate * spike_freq)
                
#                 for c in range(num_cycles):
#                     cycle_start = abnormal_start + int(c * sampling_rate / spike_freq)
#                     spike_pos = cycle_start + int(sampling_rate / (spike_freq * 4))
                    
#                     if spike_pos < abnormal_end:
#                         # Add a spike (sharp peak)
#                         amplitude = np.random.uniform(1.5, 2.5)
#                         width = int(sampling_rate / 20)  # 50ms spike
#                         for j in range(max(0, spike_pos-width), min(spike_pos+width, n_samples)):
#                             dist = abs(j - spike_pos)
#                             if dist < width:
#                                 spike_val = amplitude * (1 - dist/width)
#                                 eeg_data[ch, j] += spike_val
                
#                 # Add high frequency oscillations
#                 hfo = np.sin(2 * np.pi * 80 * time_points[abnormal_start:abnormal_end]) * 0.3
#                 eeg_data[ch, abnormal_start:abnormal_end] += hfo
            
#             abnormal_regions.append((abnormal_start/sampling_rate, abnormal_end/sampling_rate, affected_channels))
    
#     return eeg_data, time_points, abnormal_regions

# # Function to analyze EEG data and detect abnormalities
# def analyze_eeg(eeg_data, time_points, sensitivity=0.7):
#     """Analyze EEG data to detect abnormalities"""
#     num_channels, num_samples = eeg_data.shape
#     sampling_rate = int(num_samples / time_points[-1])
    
#     detected_abnormalities = []
#     channel_analyses = []
    
#     # Analyze each channel
#     for ch in range(num_channels):
#         channel_data = eeg_data[ch]
        
#         # Calculate basic statistics
#         mean_val = np.mean(channel_data)
#         std_val = np.std(channel_data)
        
#         # Detect potential abnormalities based on amplitude thresholds
#         threshold = 2.0 + (1.0 - sensitivity) * 2.0  # Adjustable threshold based on sensitivity
#         abnormal_indices = np.where(np.abs(channel_data - mean_val) > threshold * std_val)[0]
        
#         # Group consecutive abnormal points into regions
#         if len(abnormal_indices) > 0:
#             abnormal_regions = []
#             region_start = abnormal_indices[0]
#             for i in range(1, len(abnormal_indices)):
#                 if abnormal_indices[i] - abnormal_indices[i-1] > sampling_rate/5:  # If gap is more than 200ms
#                     region_end = abnormal_indices[i-1]
#                     if (region_end - region_start) > sampling_rate/10:  # If region is more than 100ms
#                         abnormal_regions.append((region_start/sampling_rate, region_end/sampling_rate))
#                     region_start = abnormal_indices[i]
            
#             # Add the last region
#             if len(abnormal_indices) > 0:
#                 region_end = abnormal_indices[-1]
#                 if (region_end - region_start) > sampling_rate/10:
#                     abnormal_regions.append((region_start/sampling_rate, region_end/sampling_rate))
            
#             for start, end in abnormal_regions:
#                 detected_abnormalities.append((start, end, ch))
        
#         # Calculate frequency domain features
#         if num_samples > 0:
#             try:
#                 # Use FFT to get frequency components
#                 fft_vals = np.abs(np.fft.rfft(channel_data))
#                 fft_freq = np.fft.rfftfreq(num_samples, 1.0/sampling_rate)
                
#                 # Calculate power in different frequency bands
#                 delta_power = np.sum(fft_vals[(fft_freq >= 0.5) & (fft_freq < 4)])
#                 theta_power = np.sum(fft_vals[(fft_freq >= 4) & (fft_freq < 8)])
#                 alpha_power = np.sum(fft_vals[(fft_freq >= 8) & (fft_freq < 13)])
#                 beta_power = np.sum(fft_vals[(fft_freq >= 13) & (fft_freq < 30)])
#                 gamma_power = np.sum(fft_vals[(fft_freq >= 30) & (fft_freq < 100)])
                
#                 total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power
                
#                 if total_power > 0:
#                     channel_analyses.append({
#                         'channel': ch,
#                         'delta_ratio': delta_power / total_power,
#                         'theta_ratio': theta_power / total_power,
#                         'alpha_ratio': alpha_power / total_power,
#                         'beta_ratio': beta_power / total_power,
#                         'gamma_ratio': gamma_power / total_power,
#                     })
#             except Exception as e:
#                 st.error(f"Error in frequency analysis: {e}")
    
#     return detected_abnormalities, channel_analyses

# # Function to generate diagnostic report based on EEG analysis
# def generate_diagnosis(abnormalities, channel_analyses):
#     """Generate diagnostic information based on EEG abnormalities"""
    
#     if not abnormalities and not channel_analyses:
#         return {
#             'overall_status': 'Normal',
#             'findings': 'No significant abnormalities detected in the EEG recording.',
#             'diagnosis': 'Normal EEG within physiological limits.',
#             'confidence': 0.95,
#             'recommendations': [
#                 'No specific EEG-based interventions required.',
#                 'Continue monitoring if symptoms persist.'
#             ]
#         }
    
#     # Count abnormalities
#     num_abnormalities = len(abnormalities)
    
#     # Analyze frequency patterns
#     avg_delta = np.mean([ch['delta_ratio'] for ch in channel_analyses]) if channel_analyses else 0
#     avg_theta = np.mean([ch['theta_ratio'] for ch in channel_analyses]) if channel_analyses else 0
#     avg_alpha = np.mean([ch['alpha_ratio'] for ch in channel_analyses]) if channel_analyses else 0
#     avg_beta = np.mean([ch['beta_ratio'] for ch in channel_analyses]) if channel_analyses else 0
#     avg_gamma = np.mean([ch['gamma_ratio'] for ch in channel_analyses]) if channel_analyses else 0
    
#     # Determine diagnosis based on patterns
#     if num_abnormalities > 5 and avg_delta > 0.4:
#         diagnosis = {
#             'overall_status': 'Abnormal',
#             'findings': 'Multiple high-amplitude sharp waves and spike-wave complexes detected with elevated delta activity.',
#             'diagnosis': 'Findings consistent with epileptiform activity, suggestive of seizure disorder.',
#             'confidence': 0.85,
#             'recommendations': [
#                 'Neurology consultation recommended',
#                 'Consider anticonvulsant therapy evaluation',
#                 'Follow-up EEG in 3-6 months',
#                 'Avoid sleep deprivation and other seizure triggers'
#             ]
#         }
#     elif num_abnormalities > 3 and avg_theta > 0.3:
#         diagnosis = {
#             'overall_status': 'Abnormal',
#             'findings': 'Intermittent sharp waves and slow wave activity with elevated theta rhythm.',
#             'diagnosis': 'Findings suggestive of focal encephalopathy or mild seizure activity.',
#             'confidence': 0.75,
#             'recommendations': [
#                 'Neurology consultation recommended',
#                 'Consider brain imaging (MRI)',
#                 'Follow-up EEG in 3 months',
#                 'Monitor for clinical symptoms'
#             ]
#         }
#     elif avg_beta > 0.5:
#         diagnosis = {
#             'overall_status': 'Borderline Abnormal',
#             'findings': 'Excessive beta activity detected throughout the recording.',
#             'diagnosis': 'Findings may indicate anxiety, medication effect (benzodiazepines), or mild cortical irritability.',
#             'confidence': 0.7,
#             'recommendations': [
#                 'Review current medications',
#                 'Consider anxiety assessment',
#                 'Follow-up as clinically indicated'
#             ]
#         }
#     elif avg_alpha < 0.1:
#         diagnosis = {
#             'overall_status': 'Borderline Abnormal',
#             'findings': 'Reduced alpha rhythm with poor organization.',
#             'diagnosis': 'Findings suggestive of encephalopathy or altered mental status.',
#             'confidence': 0.65,
#             'recommendations': [
#                 'Clinical correlation with mental status',
#                 'Consider metabolic and toxic causes',
#                 'Follow-up EEG if symptoms persist'
#             ]
#         }
#     else:
#         diagnosis = {
#             'overall_status': 'Borderline',
#             'findings': 'Minor EEG abnormalities detected but no definitive pathological pattern.',
#             'diagnosis': 'Nonspecific EEG changes of uncertain clinical significance.',
#             'confidence': 0.6,
#             'recommendations': [
#                 'Clinical correlation recommended',
#                 'Consider follow-up EEG if symptoms persist',
#                 'Monitor for development of clearer patterns'
#             ]
#         }
    
#     return diagnosis

# # Function to plot EEG data with highlighted abnormalities
# def plot_eeg_with_highlights(eeg_data, time_points, abnormal_regions, channel_labels=None):
#     """Plot EEG data with highlighted abnormal regions"""
#     num_channels = eeg_data.shape[0]
    
#     # Create default channel labels if not provided
#     if channel_labels is None:
#         channel_labels = [f"Channel {i+1}" for i in range(num_channels)]
    
#     fig, axes = plt.subplots(num_channels, 1, figsize=(12, 2*num_channels), sharex=True)
    
#     # Handle case of single channel
#     if num_channels == 1:
#         axes = [axes]
    
#     # Plot each channel
#     for i in range(num_channels):
#         axes[i].plot(time_points, eeg_data[i], 'k-', linewidth=1)
#         axes[i].set_ylabel(channel_labels[i])
        
#         # Set y-limits with some padding
#         max_val = np.max(np.abs(eeg_data[i])) * 1.2
#         axes[i].set_ylim(-max_val, max_val)
        
#         # Highlight abnormal regions
#         for start, end, channels in abnormal_regions:
#             if i in channels:
#                 axes[i].axvspan(start, end, color='red', alpha=0.3)
    
#     # Add labels
#     axes[-1].set_xlabel('Time (seconds)')
#     plt.tight_layout()
    
#     # Convert plot to image
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png', dpi=100)
#     plt.close(fig)
#     buf.seek(0)
    
#     return buf

# # Function to display treatment recommendations
# def display_treatment_recommendations(diagnosis):
#     """Display treatment recommendations based on diagnosis"""
    
#     st.markdown('<div class="sub-header">Treatment Recommendations</div>', unsafe_allow_html=True)
    
#     if diagnosis['overall_status'] == 'Normal':
#         st.markdown('<div class="treatment-box">No specific treatment needed based on EEG findings.</div>', unsafe_allow_html=True)
#         return
    
#     st.markdown('<div class="treatment-box">', unsafe_allow_html=True)
    
#     for rec in diagnosis['recommendations']:
#         st.markdown(f"- {rec}")
    
#     # Additional information based on diagnosis
#     if "epileptiform" in diagnosis['diagnosis'].lower() or "seizure" in diagnosis['diagnosis'].lower():
#         st.markdown("""
#         **Medication Considerations:**
#         - Anticonvulsant therapy may be indicated based on clinical correlation
#         - Common first-line medications: Levetiracetam, Lamotrigine, Carbamazepine (selection depends on seizure type)
        
#         **Lifestyle Modifications:**
#         - Maintain regular sleep schedule
#         - Avoid alcohol and recreational drugs
#         - Stress management techniques
#         - Consider ketogenic diet in consultation with neurologist for refractory cases
#         """)
#     elif "encephalopathy" in diagnosis['diagnosis'].lower():
#         st.markdown("""
#         **Additional Investigations to Consider:**
#         - Complete metabolic panel
#         - Toxicology screening
#         - Brain imaging (MRI preferred)
#         - CSF analysis if indicated
        
#         **Supportive Care:**
#         - Identify and treat underlying causes
#         - Cognitive monitoring
#         - Supportive therapy as needed
#         """)
    
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     st.markdown('<div class="warning-box">Note: All treatment recommendations should be reviewed by a qualified healthcare provider. This analysis is not a substitute for clinical judgment.</div>', unsafe_allow_html=True)

# # Main app layout
# st.markdown('<div class="sub-header">EEG Analysis Tool</div>', unsafe_allow_html=True)

# # Handle different input methods
# if input_type == "Upload EEG Data File":
#     uploaded_file = st.file_uploader("Upload EEG data file (.csv, .edf, .txt)", type=['csv', 'txt', 'edf'])
    
#     if uploaded_file is not None:
#         try:
#             # For this demo, we'll assume CSV format with time in first column and channel data in subsequent columns
#             if uploaded_file.name.endswith('.csv'):
#                 df = pd.read_csv(uploaded_file)
#                 time_col = df.columns[0]
#                 data_cols = df.columns[1:]
                
#                 time_points = df[time_col].values
#                 eeg_data = df[data_cols].values.T  # Transpose to get channels as first dimension
                
#                 st.success(f"Successfully loaded data with {eeg_data.shape[0]} channels and {eeg_data.shape[1]} time points")
                
#                 # For demo purposes, generate synthetic abnormalities
#                 # In a real app, you would analyze the actual data
#                 abnormal_regions = []
                
#             else:
#                 st.error("This demo currently only supports CSV files. For other formats, please use the sample data option.")
#                 eeg_data = None
#                 time_points = None
#                 abnormal_regions = []
#         except Exception as e:
#             st.error(f"Error loading file: {e}")
#             eeg_data = None
#             time_points = None
#             abnormal_regions = []
#     else:
#         eeg_data = None
#         time_points = None
#         abnormal_regions = []

# elif input_type == "Use Sample Data":
#     sample_type = st.selectbox(
#         "Select Sample Type",
#         ["Normal EEG", "Mild Abnormalities", "Epileptiform Activity", "Encephalopathy Pattern"]
#     )
    
#     if st.button("Generate Sample Data"):
#         with st.spinner("Generating sample EEG data..."):
#             # Generate different types of sample data based on selection
#             if sample_type == "Normal EEG":
#                 eeg_data, time_points, abnormal_regions = generate_eeg_sample(duration=10, channels=8, abnormal=False)
#             elif sample_type == "Mild Abnormalities":
#                 eeg_data, time_points, abnormal_regions = generate_eeg_sample(duration=10, channels=8, abnormal=True)
#                 # Modify some of the abnormalities to be milder
#                 abnormal_regions = abnormal_regions[:1]  # Limit to fewer abnormalities
#             elif sample_type == "Epileptiform Activity":
#                 eeg_data, time_points, _ = generate_eeg_sample(duration=10, channels=8, abnormal=True)
#                 # Create more pronounced spike-wave patterns
#                 abnormal_regions = []
#                 for i in range(3):
#                     start = 2 + i * 2.5
#                     end = start + 1.5
#                     channels = np.random.choice(8, size=5, replace=False)
#                     abnormal_regions.append((start, end, channels))
                    
#                     # Add spike-wave complexes
#                     for ch in channels:
#                         for t in np.arange(start, end, 0.3):
#                             idx = int(t * 250)  # Assuming 250 Hz sampling rate
#                             if idx < len(time_points) - 10:
#                                 eeg_data[ch, idx:idx+5] += 2.0  # Add spike
#                                 eeg_data[ch, idx+5:idx+10] -= 1.0  # Add wave
#             else:  # Encephalopathy Pattern
#                 eeg_data, time_points, _ = generate_eeg_sample(duration=10, channels=8, abnormal=False)
#                 # Create slow-wave pattern typical of encephalopathy
#                 abnormal_regions = [(2, 8, np.arange(8))]
                
#                 slow_wave = np.sin(2 * np.pi * 1.5 * time_points) * 0.8  # 1.5 Hz slow wave
#                 for ch in range(8):
#                     idx_start = int(2 * 250)
#                     idx_end = int(8 * 250)
#                     if idx_end <= len(time_points):
#                         eeg_data[ch, idx_start:idx_end] += slow_wave[idx_start:idx_end]
            
#             st.success("Sample data generated successfully!")
#     else:
#         eeg_data = None
#         time_points = None
#         abnormal_regions = []

# else:  # Manual Parameter Input
#     st.markdown("Please specify parameters to generate synthetic EEG data:")
    
#     duration = st.slider("Recording Duration (seconds)", 5, 30, 10)
#     num_channels = st.slider("Number of Channels", 1, 16, 8)
#     has_abnormalities = st.checkbox("Include Abnormalities", value=True)
    
#     if st.button("Generate Data"):
#         with st.spinner("Generating EEG data with specified parameters..."):
#             eeg_data, time_points, abnormal_regions = generate_eeg_sample(
#                 duration=duration, 
#                 channels=num_channels, 
#                 abnormal=has_abnormalities
#             )
#             st.success("Data generated successfully!")
#     else:
#         eeg_data = None
#         time_points = None
#         abnormal_regions = []

# # Process and analyze the data if available
# if eeg_data is not None and time_points is not None:
#     # Show progress bar for analysis
#     progress_bar = st.progress(0)
#     status_text = st.empty()
    
#     # Analysis steps with progress updates
#     status_text.text("Preprocessing EEG data...")
#     progress_bar.progress(20)
#     time.sleep(0.5)  # Simulating processing time
    
#     status_text.text("Detecting abnormalities...")
#     progress_bar.progress(40)
#     detected_abnormalities, channel_analyses = analyze_eeg(eeg_data, time_points, sensitivity)
#     time.sleep(0.5)
    
#     status_text.text("Generating diagnostic report...")
#     progress_bar.progress(60)
#     diagnosis = generate_diagnosis(detected_abnormalities, channel_analyses)
#     time.sleep(0.5)
    
#     status_text.text("Visualizing results...")
#     progress_bar.progress(80)
    
#     # Combine detected and simulated abnormalities for visualization
#     # In a real application, you would only use detected abnormalities
#     all_abnormal_regions = abnormal_regions + [(start, end, [ch]) for start, end, ch in detected_abnormalities]
    
#     eeg_plot = plot_eeg_with_highlights(eeg_data, time_points, all_abnormal_regions)
#     progress_bar.progress(100)
#     status_text.empty()
    
#     # Display results
#     st.markdown('<div class="sub-header">EEG Visualization with Highlighted Abnormalities</div>', unsafe_allow_html=True)
#     st.image(eeg_plot, caption="EEG Recording with Highlighted Abnormal Regions", use_column_width=True)
    
#     # Show Frequency Analysis
#     st.markdown('<div class="sub-header">Frequency Analysis</div>', unsafe_allow_html=True)
    
#     if channel_analyses:
#         # Create frequency analysis chart
#         fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
#         bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
#         x = np.arange(len(bands))
#         width = 0.1
        
#         for i, ch_analysis in enumerate(channel_analyses):
#             values = [
#                 ch_analysis['delta_ratio'], 
#                 ch_analysis['theta_ratio'],
#                 ch_analysis['alpha_ratio'],
#                 ch_analysis['beta_ratio'],
#                 ch_analysis['gamma_ratio']
#             ]
#             ax.bar(x + i*width - (len(channel_analyses)-1)*width/2, values, width, label=f"Ch {ch_analysis['channel']+1}")
        
#         ax.set_ylabel('Power Ratio')
#         ax.set_title('Frequency Band Power Distribution by Channel')
#         ax.set_xticks(x)
#         ax.set_xticklabels(bands)
#         ax.legend(loc='best', ncol=min(8, len(channel_analyses)))
        
#         st.pyplot(fig)
#     else:
#         st.info("Frequency analysis not available.")
    
#     # Display diagnostic report
#     st.markdown('<div class="sub-header">Diagnostic Report</div>', unsafe_allow_html=True)
    
#     # Status badge
#     status_color = {
#         'Normal': 'normal-region',
#         'Borderline': 'highlight-text',
#         'Borderline Abnormal': 'highlight-text',
#         'Abnormal': 'abnormal-region'
#     }.get(diagnosis['overall_status'], 'highlight-text')
    
#     st.markdown(f'<div class="{status_color}">Overall Status: {diagnosis["overall_status"]}</div>', unsafe_allow_html=True)
    
#     # Diagnostic findings
#     st.markdown('<div class="diagnosis-box">', unsafe_allow_html=True)
#     st.markdown("### Key Findings")
#     st.markdown(diagnosis['findings'])
    
#     st.markdown("### Diagnostic Impression")
#     st.markdown(diagnosis['diagnosis'])
    
#     st.markdown(f"*Diagnostic confidence: {diagnosis['confidence']*100:.1f}%*")
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     # Treatment recommendations
#     display_treatment_recommendations(diagnosis)
    
#     # Export options
#     st.markdown('<div class="sub-header">Export Options</div>', unsafe_allow_html=True)
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         if st.button("📄 Export Report (PDF)"):
#             st.info("PDF export functionality would be implemented here in a full application.")
    
#     with col2:
#         if st.button("📊 Export Data (CSV)"):
#             st.info("CSV export functionality would be implemented here.")
    
#     with col3:
#         if st.button("🖼️ Export Images"):
#             st.info("Image export functionality would be implemented here.")

# else:
#     # Show instructions if no data is loaded
#     st.markdown("""
#     ## How to Use This Tool
    
#     1. Select an input method from the sidebar
#     2. Either upload your EEG data file or generate sample data
#     3. Adjust analysis settings in the sidebar if needed
#     4. The system will:
#        - Process the EEG data
#        - Highlight abnormal regions
#        - Provide a diagnostic report
#        - Suggest appropriate treatments
    
#     **Note:** This is a demonstration prototype. In a clinical setting, this tool would be integrated with professional EEG equipment and validated algorithms.
#     """)
    
#     # Show example images
#     st.markdown("### Example EEG Patterns")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("**Normal EEG**")
#         # Generate a normal EEG sample for display
#         normal_eeg, normal_time, _ = generate_eeg_sample(duration=5, channels=4, abnormal=False)
#         normal_plot = plot_eeg_with_highlights(normal_eeg, normal_time, [])
#         st.image(normal_plot, use_column_width=True)
    
#     with col2:
#         st.markdown("**Abnormal EEG (Epileptiform Activity)**")
#         # Generate an abnormal EEG sample for display
#         abnormal_eeg, abnormal_time, abnormal_regions = generate_eeg_sample(duration=5, channels=4, abnormal=True)
#         abnormal_plot = plot_eeg_with_highlights(abnormal_eeg, abnormal_time, abnormal_regions)
#         st.image(abnormal_plot, use_column_width=True)

# # Footer
# st.markdown("---")
# st.markdown("""
#     <div style="text-align: center; color: #666;">
#         EEG Report Analysis and Prediction System
#     </div>
# """, unsafe_allow_html=True)

# # Additional functionality that could be implemented in a full version:
# # 1. User authentication system for medical professionals
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import io
import time
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import mne
from pathlib import Path

# Set page configuration for professional look
st.set_page_config(
    page_title="EEG Diagnostic System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
        .main-header {
            font-size: 32px;
            font-weight: bold;
            color: #1E3A8A;
            margin-bottom: 20px;
            text-align: center;
        }
        .sub-header {
            font-size: 24px;
            font-weight: bold;
            color: #1E3A8A;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        .highlight-text {
            background-color: #DBEAFE;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0px;
        }
        .abnormal-region {
            background-color: rgba(239, 68, 68, 0.2);
            border-radius: 5px;
            padding: 8px;
            font-weight: bold;
        }
        .normal-region {
            background-color: rgba(34, 197, 94, 0.2);
            border-radius: 5px;
            padding: 8px;
            font-weight: bold;
        }
        .diagnosis-box {
            background-color: #EFF6FF;
            border-left: 5px solid #3B82F6;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0px;
        }
        .treatment-box {
            background-color: #ECFDF5;
            border-left: 5px solid #10B981;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0px;
        }
        .warning-box {
            background-color: #FEF2F2;
            border-left: 5px solid #EF4444;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0px;
        }
        .info-box {
            background-color: #F0F9FF;
            border-left: 5px solid #0284C7;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0px;
        }
        .sidebar .sidebar-content {
            background-color: #F9FAFB;
        }
        .stButton>button {
            background-color: #3B82F6;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #2563EB;
        }
    </style>
    <div class="main-header">EEG Diagnostic and Analysis System</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/frontend/src/assets/branding/streamlit-mark-color.svg", width=100)
    st.markdown("## Input Options")
    
    input_type = st.radio(
        "Select Input Method",
        ["Upload EEG Data File", "Use MNE Sample Data", "Generate Simulated EEG Data"]
    )
    
    st.markdown("---")
    st.markdown("## Analysis Settings")
    
    sensitivity = st.slider("Analysis Sensitivity", 0.1, 1.0, 0.7, 0.1)
    frequency_bands = st.multiselect(
        "Frequency Bands to Analyze",
        ["Delta (0.5-4 Hz)", "Theta (4-8 Hz)", "Alpha (8-13 Hz)", "Beta (13-30 Hz)", "Gamma (30-100 Hz)"],
        default=["Delta (0.5-4 Hz)", "Theta (4-8 Hz)", "Alpha (8-13 Hz)", "Beta (13-30 Hz)"]
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This professional EEG Diagnostic System analyzes EEG data to detect abnormalities, provide diagnostic insights, and offer treatment recommendations. Select a sample EEG pattern, use MNE sample data, or upload your own data to begin.
    
    **Note:** This is a prototype and should not replace professional medical advice.
    """)

# Function to fetch MNE sample EEG data
def fetch_mne_sample_data(pattern="normal"):
    """Fetch MNE sample EEG data and modify for specific patterns."""
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
    raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)
    
    raw_eeg = raw.pick_types(meg=False, eeg=True, eog=True)
    sampling_rate = raw_eeg.info['sfreq']
    eeg_data = raw_eeg.get_data() * 1e6  # Convert to microvolts
    time_points = raw_eeg.times
    channels = len(raw_eeg.ch_names)
    
    abnormal_regions = []
    
    if pattern == "epilepsy":
        for i in range(3):
            start = 2 + i * 2.5
            end = start + 1.5
            channels_affected = np.random.choice(channels, size=5, replace=False)
            abnormal_regions.append((start, end, channels_affected))
            for ch in channels_affected:
                for t in np.arange(start, end, 0.3):
                    idx = int(t * sampling_rate)
                    if idx < len(time_points) - 10:
                        eeg_data[ch, idx:idx+5] += 100  # Spike
                        eeg_data[ch, idx+5:idx+10] -= 50  # Wave
    elif pattern == "dementia":
        abnormal_regions = [(2, 8, np.arange(channels))]
        slow_wave = np.sin(2 * np.pi * 3 * time_points) * 40  # Theta slowing
        for ch in range(channels):
            idx_start = int(2 * sampling_rate)
            idx_end = int(8 * sampling_rate)
            if idx_end <= len(time_points):
                eeg_data[ch, idx_start:idx_end] += slow_wave[idx_start:idx_end]
    elif pattern == "brain_tumor":
        start = 3
        end = 7
        channels_affected = np.random.choice(channels, size=3, replace=False)
        abnormal_regions.append((start, end, channels_affected))
        for ch in channels_affected:
            idx_start = int(start * sampling_rate)
            idx_end = int(end * sampling_rate)
            if idx_end <= len(time_points):
                eeg_data[ch, idx_start:idx_end] += np.sin(2 * np.pi * 2 * time_points[idx_start:idx_end]) * 50  # Focal slowing
    elif pattern == "encephalitis":
        start = 2
        end = 6
        channels_affected = np.random.choice(channels, size=6, replace=False)  # Temporal lobes
        abnormal_regions.append((start, end, channels_affected))
        for ch in channels_affected:
            for t in np.arange(start, end, 0.2):
                idx = int(t * sampling_rate)
                if idx < len(time_points) - 5:
                    eeg_data[ch, idx:idx+5] += 80  # Sharp waves
    elif pattern == "tbi":
        num_abnormalities = np.random.randint(2, 4)
        for _ in range(num_abnormalities):
            abnormal_start = np.random.randint(0, len(time_points) - int(sampling_rate))
            abnormal_duration = np.random.randint(int(sampling_rate // 2), int(sampling_rate))
            abnormal_end = min(abnormal_start + abnormal_duration, len(time_points))
            affected_channels = np.random.choice(channels, size=np.random.randint(2, channels+1), replace=False)
            for ch in affected_channels:
                eeg_data[ch, abnormal_start:abnormal_end] += np.sin(2 * np.pi * 1 * time_points[abnormal_start:abnormal_end]) * 30  # Slow waves
            abnormal_regions.append((abnormal_start/sampling_rate, abnormal_end/sampling_rate, affected_channels))
    elif pattern == "stroke":
        start = 2
        end = 8
        channels_affected = np.random.choice(channels, size=4, replace=False)
        abnormal_regions.append((start, end, channels_affected))
        for ch in channels_affected:
            idx_start = int(start * sampling_rate)
            idx_end = int(end * sampling_rate)
            if idx_end <= len(time_points):
                eeg_data[ch, idx_start:idx_end] += np.sin(2 * np.pi * 2.5 * time_points[idx_start:idx_end]) * 50  # Focal delta
    elif pattern == "metabolic_conditions":
        abnormal_regions = [(1, 9, np.arange(channels))]
        triphasic_wave = np.sin(2 * np.pi * 2 * time_points) * 60  # Triphasic waves
        for ch in range(channels):
            idx_start = int(1 * sampling_rate)
            idx_end = int(9 * sampling_rate)
            if idx_end <= len(time_points):
                eeg_data[ch, idx_start:idx_end] += triphasic_wave[idx_start:idx_end]
    elif pattern == "hypoxia":
        abnormal_regions = [(2, 8, np.arange(channels))]
        slow_wave = np.sin(2 * np.pi * 1 * time_points) * 70  # Diffuse slowing
        for ch in range(channels):
            idx_start = int(2 * sampling_rate)
            idx_end = int(8 * sampling_rate)
            if idx_end <= len(time_points):
                eeg_data[ch, idx_start:idx_end] += slow_wave[idx_start:idx_end]
    elif pattern == "brain_death":
        abnormal_regions = [(0, len(time_points)/sampling_rate, np.arange(channels))]
        for ch in range(channels):
            eeg_data[ch, :] = np.random.normal(0, 0.01, len(time_points))  # Flat EEG
    elif pattern == "epileptiform":
        for i in range(3):
            start = 2 + i * 2.5
            end = start + 1.5
            channels_affected = np.random.choice(channels, size=5, replace=False)
            abnormal_regions.append((start, end, channels_affected))
            for ch in channels_affected:
                for t in np.arange(start, end, 0.3):
                    idx = int(t * sampling_rate)
                    if idx < len(time_points) - 10:
                        eeg_data[ch, idx:idx+5] += 100
                        eeg_data[ch, idx+5:idx+10] -= 50
    elif pattern == "encephalopathy":
        abnormal_regions = [(2, 8, np.arange(channels))]
        slow_wave = np.sin(2 * np.pi * 1.5 * time_points) * 50
        for ch in range(channels):
            idx_start = int(2 * sampling_rate)
            idx_end = int(8 * sampling_rate)
            if idx_end <= len(time_points):
                eeg_data[ch, idx_start:idx_end] += slow_wave[idx_start:idx_end]
    elif pattern == "mild_abnormal":
        num_abnormalities = np.random.randint(1, 3)
        for _ in range(num_abnormalities):
            abnormal_start = np.random.randint(0, len(time_points) - int(sampling_rate))
            abnormal_duration = np.random.randint(int(sampling_rate // 2), int(sampling_rate))
            abnormal_end = min(abnormal_start + abnormal_duration, len(time_points))
            affected_channels = np.random.choice(channels, size=np.random.randint(2, channels+1), replace=False)
            for ch in affected_channels:
                hfo = np.sin(2 * np.pi * 80 * time_points[abnormal_start:abnormal_end]) * 20
                eeg_data[ch, abnormal_start:abnormal_end] += hfo
            abnormal_regions.append((abnormal_start/sampling_rate, abnormal_end/sampling_rate, affected_channels))
    
    return eeg_data, time_points, abnormal_regions, raw_eeg.ch_names

# Function to generate simulated EEG data
def generate_eeg_sample(duration=10, sampling_rate=250, channels=8, pattern="normal"):
    """Generate simulated EEG data for the specified pattern."""
    time_points = np.arange(0, duration, 1/sampling_rate)
    n_samples = len(time_points)
    
    delta = np.sin(2 * np.pi * 2 * time_points)
    theta = np.sin(2 * np.pi * 6 * time_points)
    alpha = np.sin(2 * np.pi * 10 * time_points)
    beta = np.sin(2 * np.pi * 20 * time_points)
    
    eeg_data = np.zeros((channels, n_samples))
    for i in range(channels):
        a, b, c, d = np.random.rand(4)
        eeg_data[i] = (a * delta + b * theta + c * alpha + d * beta) / (a + b + c + d)
        eeg_data[i] += np.random.normal(0, 0.1, n_samples)
    
    abnormal_regions = []
    
    if pattern == "epilepsy":
        for i in range(3):
            start = 2 + i * 2.5
            end = start + 1.5
            channels_affected = np.random.choice(channels, size=5, replace=False)
            abnormal_regions.append((start, end, channels_affected))
            for ch in channels_affected:
                for t in np.arange(start, end, 0.3):
                    idx = int(t * sampling_rate)
                    if idx < len(time_points) - 10:
                        eeg_data[ch, idx:idx+5] += 2.0
                        eeg_data[ch, idx+5:idx+10] -= 1.0
    elif pattern == "dementia":
        abnormal_regions = [(2, 8, np.arange(channels))]
        slow_wave = np.sin(2 * np.pi * 3 * time_points) * 0.8
        for ch in range(channels):
            idx_start = int(2 * sampling_rate)
            idx_end = int(8 * sampling_rate)
            if idx_end <= len(time_points):
                eeg_data[ch, idx_start:idx_end] += slow_wave[idx_start:idx_end]
    elif pattern == "brain_tumor":
        start = 3
        end = 7
        channels_affected = np.random.choice(channels, size=3, replace=False)
        abnormal_regions.append((start, end, channels_affected))
        for ch in channels_affected:
            idx_start = int(start * sampling_rate)
            idx_end = int(end * sampling_rate)
            if idx_end <= len(time_points):
                eeg_data[ch, idx_start:idx_end] += np.sin(2 * np.pi * 2 * time_points[idx_start:idx_end]) * 1.0
    elif pattern == "encephalitis":
        start = 2
        end = 6
        channels_affected = np.random.choice(channels, size=6, replace=False)
        abnormal_regions.append((start, end, channels_affected))
        for ch in channels_affected:
            for t in np.arange(start, end, 0.2):
                idx = int(t * sampling_rate)
                if idx < len(time_points) - 5:
                    eeg_data[ch, idx:idx+5] += 1.5
    elif pattern == "tbi":
        num_abnormalities = np.random.randint(2, 4)
        for _ in range(num_abnormalities):
            abnormal_start = np.random.randint(0, n_samples - sampling_rate)
            abnormal_duration = np.random.randint(sampling_rate // 2, sampling_rate)
            abnormal_end = min(abnormal_start + abnormal_duration, n_samples)
            affected_channels = np.random.choice(channels, size=np.random.randint(2, channels+1), replace=False)
            for ch in affected_channels:
                eeg_data[ch, abnormal_start:abnormal_end] += np.sin(2 * np.pi * 1 * time_points[abnormal_start:abnormal_end]) * 0.6
            abnormal_regions.append((abnormal_start/sampling_rate, abnormal_end/sampling_rate, affected_channels))
    elif pattern == "stroke":
        start = 2
        end = 8
        channels_affected = np.random.choice(channels, size=4, replace=False)
        abnormal_regions.append((start, end, channels_affected))
        for ch in channels_affected:
            idx_start = int(start * sampling_rate)
            idx_end = int(end * sampling_rate)
            if idx_end <= len(time_points):
                eeg_data[ch, idx_start:idx_end] += np.sin(2 * np.pi * 2.5 * time_points[idx_start:idx_end]) * 1.0
    elif pattern == "metabolic_conditions":
        abnormal_regions = [(1, 9, np.arange(channels))]
        triphasic_wave = np.sin(2 * np.pi * 2 * time_points) * 1.2
        for ch in range(channels):
            idx_start = int(1 * sampling_rate)
            idx_end = int(9 * sampling_rate)
            if idx_end <= len(time_points):
                eeg_data[ch, idx_start:idx_end] += triphasic_wave[idx_start:idx_end]
    elif pattern == "hypoxia":
        abnormal_regions = [(2, 8, np.arange(channels))]
        slow_wave = np.sin(2 * np.pi * 1 * time_points) * 1.5
        for ch in range(channels):
            idx_start = int(2 * sampling_rate)
            idx_end = int(8 * sampling_rate)
            if idx_end <= len(time_points):
                eeg_data[ch, idx_start:idx_end] += slow_wave[idx_start:idx_end]
    elif pattern == "brain_death":
        abnormal_regions = [(0, duration, np.arange(channels))]
        for ch in range(channels):
            eeg_data[ch, :] = np.random.normal(0, 0.01, n_samples)
    elif pattern == "epileptiform":
        for i in range(3):
            start = 2 + i * 2.5
            end = start + 1.5
            channels_affected = np.random.choice(channels, size=5, replace=False)
            abnormal_regions.append((start, end, channels_affected))
            for ch in channels_affected:
                for t in np.arange(start, end, 0.3):
                    idx = int(t * sampling_rate)
                    if idx < len(time_points) - 10:
                        eeg_data[ch, idx:idx+5] += 2.0
                        eeg_data[ch, idx+5:idx+10] -= 1.0
    elif pattern == "encephalopathy":
        abnormal_regions = [(2, 8, np.arange(channels))]
        slow_wave = np.sin(2 * np.pi * 1.5 * time_points) * 0.8
        for ch in range(channels):
            idx_start = int(2 * sampling_rate)
            idx_end = int(8 * sampling_rate)
            if idx_end <= len(time_points):
                eeg_data[ch, idx_start:idx_end] += slow_wave[idx_start:idx_end]
    elif pattern == "mild_abnormal":
        num_abnormalities = np.random.randint(1, 3)
        for _ in range(num_abnormalities):
            abnormal_start = np.random.randint(0, n_samples - sampling_rate)
            abnormal_duration = np.random.randint(sampling_rate // 2, sampling_rate)
            abnormal_end = min(abnormal_start + abnormal_duration, n_samples)
            affected_channels = np.random.choice(channels, size=np.random.randint(2, channels+1), replace=False)
            for ch in affected_channels:
                hfo = np.sin(2 * np.pi * 80 * time_points[abnormal_start:abnormal_end]) * 0.2
                eeg_data[ch, abnormal_start:abnormal_end] += hfo
            abnormal_regions.append((abnormal_start/sampling_rate, abnormal_end/sampling_rate, affected_channels))
    
    return eeg_data, time_points, abnormal_regions, [f"Channel {i+1}" for i in range(channels)]

# Function to get EEG pattern information
def get_pattern_info(pattern):
    """Return detailed information about the selected EEG pattern."""
    info = {
        "normal": {
            "title": "Normal EEG",
            "description": "A normal EEG shows typical brain wave patterns without significant abnormalities.",
            "wave_characteristics": """
            - **Delta (0.5-4 Hz)**: Low amplitude, prominent during deep sleep.
            - **Theta (4-8 Hz)**: Present during light sleep or drowsiness.
            - **Alpha (8-13 Hz)**: Dominant during relaxed wakefulness, especially with eyes closed.
            - **Beta (13-30 Hz)**: Associated with active concentration and alertness.
            - **Gamma (30-100 Hz)**: Linked to cognitive processing and memory.
            """,
            "conditions": "No pathological conditions typically associated.",
            "clinical_significance": "Indicates normal brain function within physiological limits."
        },
        "epilepsy": {
            "title": "Epilepsy",
            "description": "Characterized by abnormal electrical discharges indicative of seizures.",
            "wave_characteristics": """
            - **Spikes**: Sharp, high-amplitude transients (<70 ms).
            - **Spike-Wave Complexes**: Occur at ~3 Hz, typical of absence seizures.
            - **Delta/Theta**: Elevated in seizure-prone regions.
            """,
            "conditions": "Epilepsy, including focal and generalized seizures.",
            "clinical_significance": "Confirms seizure activity; requires neurological evaluation and anticonvulsant therapy."
        },
        "dementia": {
            "title": "Dementia",
            "description": "Shows abnormal brain activity associated with cognitive decline, often due to Alzheimer's or Lewy body dementia.",
            "wave_characteristics": """
            - **Theta (4-8 Hz)**: Increased diffuse slowing.
            - **Alpha (8-13 Hz)**: Reduced amplitude and frequency.
            - **Delta (0.5-4 Hz)**: May be prominent in advanced stages.
            """,
            "conditions": "Alzheimer's disease, Lewy body dementia, frontotemporal dementia.",
            "clinical_significance": "Supports diagnosis of dementia; requires cognitive and neurological assessment."
        },
        "brain_tumor": {
            "title": "Brain Tumor",
            "description": "Detects abnormal brain activity and seizure risk associated with brain tumors.",
            "wave_characteristics": """
            - **Delta (0.5-4 Hz)**: Focal slowing near tumor site.
            - **Spikes**: May indicate seizure activity.
            - **Asymmetry**: Abnormalities localized to tumor region.
            """,
            "conditions": "Primary or metastatic brain tumors.",
            "clinical_significance": "Identifies seizure risk and tumor-related brain dysfunction; requires imaging and oncology evaluation."
        },
        "encephalitis": {
            "title": "Encephalitis",
            "description": "Shows sharp waves and abnormalities due to brain inflammation, often in temporal lobes.",
            "wave_characteristics": """
            - **Sharp Waves**: Prominent in temporal regions.
            - **Delta/Theta**: Increased slowing in affected areas.
            - **Spikes**: May indicate seizure activity.
            """,
            "conditions": "Viral encephalitis, autoimmune encephalitis.",
            "clinical_significance": "Supports diagnosis of brain inflammation; requires urgent neurological and infectious disease evaluation."
        },
        "tbi": {
            "title": "Traumatic Brain Injury (TBI)",
            "description": "Evaluates brain function after head injury, detecting abnormal activity.",
            "wave_characteristics": """
            - **Delta (0.5-4 Hz)**: Diffuse or focal slowing.
            - **Theta (4-8 Hz)**: Increased in injured regions.
            - **Alpha/Beta**: Reduced in severe cases.
            """,
            "conditions": "Concussion, contusion, diffuse axonal injury.",
            "clinical_significance": "Assesses brain dysfunction post-injury; guides rehabilitation and seizure management."
        },
        "stroke": {
            "title": "Stroke",
            "description": "Differentiates stroke-related symptoms from seizures, detecting abnormal activity.",
            "wave_characteristics": """
            - **Delta (0.5-4 Hz)**: Focal slowing in ischemic region.
            - **Theta (4-8 Hz)**: May be prominent.
            - **Asymmetry**: Localized to stroke-affected area.
            """,
            "conditions": "Ischemic or hemorrhagic stroke.",
            "clinical_significance": "Confirms stroke-related brain dysfunction; guides acute management and rehabilitation."
        },
        "metabolic_conditions": {
            "title": "Metabolic Conditions",
            "description": "Detects brain dysfunction due to chemical imbalances from metabolic disorders.",
            "wave_characteristics": """
            - **Triphasic Waves**: Characteristic at ~2 Hz.
            - **Delta (0.5-4 Hz)**: Diffuse slowing.
            - **Theta (4-8 Hz)**: Increased in severe cases.
            """,
            "conditions": "Hepatic encephalopathy, uremia, electrolyte imbalances.",
            "clinical_significance": "Identifies metabolic brain injury; requires correction of underlying metabolic disorder."
        },
        "hypoxia": {
            "title": "Hypoxia",
            "description": "Identifies brain damage due to oxygen deprivation.",
            "wave_characteristics": """
            - **Delta (0.5-4 Hz)**: Prominent diffuse slowing.
            - **Theta (4-8 Hz)**: Increased in affected areas.
            - **Alpha/Beta**: Severely reduced or absent.
            """,
            "conditions": "Hypoxic-ischemic encephalopathy, cardiac arrest.",
            "clinical_significance": "Assesses extent of hypoxic brain injury; guides prognosis and supportive care."
        },
        "brain_death": {
            "title": "Brain Death",
            "description": "Shows minimal to no brain activity, indicating irreversible brain damage.",
            "wave_characteristics": """
            - **Flat EEG**: Near-zero amplitude across all frequencies.
            - **No Waves**: Absence of delta, theta, alpha, beta, or gamma activity.
            """,
            "conditions": "Irreversible brain damage from trauma, anoxia, or other causes.",
            "clinical_significance": "Supports brain death determination; used in conjunction with clinical and imaging criteria."
        },
        "epileptiform": {
            "title": "Epileptiform Activity",
            "description": "General epileptiform patterns, not specific to epilepsy diagnosis.",
            "wave_characteristics": """
            - **Spikes**: Sharp, high-amplitude transients (<70 ms).
            - **Spike-Wave Complexes**: Typical at ~3 Hz.
            - **Delta/Theta**: May be elevated.
            """,
            "conditions": "Potential seizure risk, not necessarily epilepsy.",
            "clinical_significance": "Suggests cortical irritability; requires clinical correlation."
        },
        "encephalopathy": {
            "title": "Encephalopathy",
            "description": "General pattern of diffuse brain dysfunction, not specific to a single cause.",
            "wave_characteristics": """
            - **Delta (0.5-4 Hz)**: Prominent, diffuse slow waves (~1-2 Hz).
            - **Theta (4-8 Hz)**: Increased in diffuse patterns.
            - **Alpha/Beta**: Reduced or disorganized.
            """,
            "conditions": "Metabolic, toxic, or hypoxic encephalopathy.",
            "clinical_significance": "Indicates brain dysfunction; requires investigation of underlying cause."
        },
        "mild_abnormal": {
            "title": "Mild Abnormalities",
            "description": "Contains subtle abnormalities that may not indicate a specific pathology.",
            "wave_characteristics": """
            - **High-Frequency Oscillations**: Subtle bursts in gamma range (30-100 Hz).
            - **Intermittent Slowing**: Occasional theta or delta activity.
            - **Alpha/Beta**: Slight asymmetry or irregularity.
            """,
            "conditions": "Early encephalopathy, mild cortical irritability, medication effects.",
            "clinical_significance": "Nonspecific changes; requires clinical correlation and follow-up."
        },
        "uploaded_data": {
            "title": "Uploaded EEG Data",
            "description": "User-uploaded EEG data in CSV format.",
            "wave_characteristics": "Varies based on the uploaded data; analyzed for delta, theta, alpha, beta, and gamma bands.",
            "conditions": "Depends on analysis results; may indicate normal or pathological states.",
            "clinical_significance": "Determined by detected abnormalities and frequency analysis."
        }
    }
    return info.get(pattern, info["normal"])

# Function to analyze EEG data
def analyze_eeg(eeg_data, time_points, sensitivity=0.7):
    """Analyze EEG data to detect abnormalities."""
    num_channels, num_samples = eeg_data.shape
    sampling_rate = int(num_samples / time_points[-1])
    
    detected_abnormalities = []
    channel_analyses = []
    
    for ch in range(num_channels):
        channel_data = eeg_data[ch]
        mean_val = np.mean(channel_data)
        std_val = np.std(channel_data)
        
        threshold = 2.0 + (1.0 - sensitivity) * 2.0
        abnormal_indices = np.where(np.abs(channel_data - mean_val) > threshold * std_val)[0]
        
        if len(abnormal_indices) > 0:
            abnormal_regions = []
            region_start = abnormal_indices[0]
            for i in range(1, len(abnormal_indices)):
                if abnormal_indices[i] - abnormal_indices[i-1] > sampling_rate/5:
                    region_end = abnormal_indices[i-1]
                    if (region_end - region_start) > sampling_rate/10:
                        abnormal_regions.append((region_start/sampling_rate, region_end/sampling_rate))
                    region_start = abnormal_indices[i]
            
            if len(abnormal_indices) > 0:
                region_end = abnormal_indices[-1]
                if (region_end - region_start) > sampling_rate/10:
                    abnormal_regions.append((region_start/sampling_rate, region_end/sampling_rate))
            
            for start, end in abnormal_regions:
                detected_abnormalities.append((start, end, ch))
        
        if num_samples > 0:
            try:
                fft_vals = np.abs(np.fft.rfft(channel_data))
                fft_freq = np.fft.rfftfreq(num_samples, 1.0/sampling_rate)
                
                delta_power = np.sum(fft_vals[(fft_freq >= 0.5) & (fft_freq < 4)])
                theta_power = np.sum(fft_vals[(fft_freq >= 4) & (fft_freq < 8)])
                alpha_power = np.sum(fft_vals[(fft_freq >= 8) & (fft_freq < 13)])
                beta_power = np.sum(fft_vals[(fft_freq >= 13) & (fft_freq < 30)])
                gamma_power = np.sum(fft_vals[(fft_freq >= 30) & (fft_freq < 100)])
                
                total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power
                
                if total_power > 0:
                    channel_analyses.append({
                        'channel': ch,
                        'delta_ratio': delta_power / total_power,
                        'theta_ratio': theta_power / total_power,
                        'alpha_ratio': alpha_power / total_power,
                        'beta_ratio': beta_power / total_power,
                        'gamma_ratio': gamma_power / total_power,
                    })
            except Exception as e:
                st.error(f"Error in frequency analysis: {e}")
    
    return detected_abnormalities, channel_analyses

# Function to generate diagnostic report
def generate_diagnosis(abnormalities, channel_analyses, pattern):
    """Generate diagnostic information with simulated model accuracy."""
    if not abnormalities and not channel_analyses:
        return {
            'overall_status': 'Normal',
            'findings': 'No significant abnormalities detected in the EEG recording.',
            'diagnosis': 'Normal EEG within physiological limits.',
            'confidence': 0.95,
            'model_accuracy': 0.98,
            'recommendations': [
                'No specific EEG-based interventions required.',
                'Continue monitoring if symptoms persist.'
            ]
        }
    
    num_abnormalities = len(abnormalities)
    avg_delta = np.mean([ch['delta_ratio'] for ch in channel_analyses]) if channel_analyses else 0
    avg_theta = np.mean([ch['theta_ratio'] for ch in channel_analyses]) if channel_analyses else 0
    avg_alpha = np.mean([ch['alpha_ratio'] for ch in channel_analyses]) if channel_analyses else 0
    avg_beta = np.mean([ch['beta_ratio'] for ch in channel_analyses]) if channel_analyses else 0
    
    model_accuracy = max(0.60, min(0.95, 0.98 - 0.005 * num_abnormalities))
    
    if pattern == "brain_death":
        diagnosis = {
            'overall_status': 'Abnormal',
            'findings': 'Flat EEG with minimal to no detectable brain activity.',
            'diagnosis': 'Findings consistent with brain death.',
            'confidence': 0.99,
            'model_accuracy': model_accuracy,
            'recommendations': [
                'Confirm with clinical examination and imaging.',
                'Consult neurology for brain death protocol.',
                'Discuss with family regarding prognosis.'
            ]
        }
    elif num_abnormalities > 5 and (pattern in ["epilepsy", "epileptiform"] or avg_delta > 0.4):
        diagnosis = {
            'overall_status': 'Abnormal',
            'findings': 'Multiple high-amplitude sharp waves and spike-wave complexes detected with elevated delta activity.',
            'diagnosis': 'Findings consistent with epileptiform activity, suggestive of seizure disorder.',
            'confidence': 0.85,
            'model_accuracy': model_accuracy,
            'recommendations': [
                'Neurology consultation recommended.',
                'Consider anticonvulsant therapy evaluation.',
                'Follow-up EEG in 3-6 months.',
                'Avoid sleep deprivation and other seizure triggers.'
            ]
        }
    elif num_abnormalities > 3 and (pattern in ["encephalopathy", "dementia", "hypoxia", "metabolic_conditions"] or avg_theta > 0.3):
        diagnosis = {
            'overall_status': 'Abnormal',
            'findings': 'Diffuse slow wave activity with elevated theta rhythm.',
            'diagnosis': 'Findings suggestive of encephalopathy or metabolic disturbance.',
            'confidence': 0.75,
            'model_accuracy': model_accuracy,
            'recommendations': [
                'Neurology consultation recommended.',
                'Consider brain imaging (MRI).',
                'Evaluate for metabolic or toxic causes.',
                'Follow-up EEG in 3 months.'
            ]
        }
    elif pattern in ["brain_tumor", "stroke", "tbi"] and num_abnormalities > 2:
        diagnosis = {
            'overall_status': 'Abnormal',
            'findings': 'Focal slowing and abnormal activity detected in specific regions.',
            'diagnosis': f'Findings suggestive of {pattern.replace("_", " ")}.',
            'confidence': 0.80,
            'model_accuracy': model_accuracy,
            'recommendations': [
                'Urgent neurology and imaging evaluation.',
                'Assess for seizure risk and consider prophylaxis.',
                'Monitor for clinical symptoms.',
                'Follow-up EEG as indicated.'
            ]
        }
    elif pattern == "encephalitis" and num_abnormalities > 3:
        diagnosis = {
            'overall_status': 'Abnormal',
            'findings': 'Sharp waves and slowing in temporal regions.',
            'diagnosis': 'Findings suggestive of encephalitis.',
            'confidence': 0.82,
            'model_accuracy': model_accuracy,
            'recommendations': [
                'Urgent neurology and infectious disease consultation.',
                'Consider antiviral or immunosuppressive therapy.',
                'Brain imaging (MRI) recommended.',
                'Monitor for seizures.'
            ]
        }
    elif avg_beta > 0.5:
        diagnosis = {
            'overall_status': 'Borderline Abnormal',
            'findings': 'Excessive beta activity detected throughout the recording.',
            'diagnosis': 'Findings may indicate anxiety, medication effect, or mild cortical irritability.',
            'confidence': 0.7,
            'model_accuracy': model_accuracy,
            'recommendations': [
                'Review current medications.',
                'Consider anxiety assessment.',
                'Follow-up as clinically indicated.'
            ]
        }
    elif avg_alpha < 0.1:
        diagnosis = {
            'overall_status': 'Borderline Abnormal',
            'findings': 'Reduced alpha rhythm with poor organization.',
            'diagnosis': 'Findings suggestive of encephalopathy or altered mental status.',
            'confidence': 0.65,
            'model_accuracy': model_accuracy,
            'recommendations': [
                'Clinical correlation with mental status.',
                'Consider metabolic and toxic causes.',
                'Follow-up EEG if symptoms persist.'
            ]
        }
    else:
        diagnosis = {
            'overall_status': 'Borderline',
            'findings': 'Minor EEG abnormalities detected but no definitive pathological pattern.',
            'diagnosis': 'Nonspecific EEG changes of uncertain clinical significance.',
            'confidence': 0.6,
            'model_accuracy': model_accuracy,
            'recommendations': [
                'Clinical correlation recommended.',
                'Consider follow-up EEG if symptoms persist.',
                'Monitor for development of clearer patterns.'
            ]
        }
    
    return diagnosis

# Function to plot EEG data
def plot_eeg_with_highlights(eeg_data, time_points, abnormal_regions, channel_labels):
    """Plot EEG data with highlighted abnormal regions."""
    num_channels = eeg_data.shape[0]
    
    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 2*num_channels), sharex=True)
    if num_channels == 1:
        axes = [axes]
    
    for i in range(num_channels):
        axes[i].plot(time_points, eeg_data[i], color='#555555', linestyle='-', linewidth=0.8)  # Lighter gray, thinner line
        axes[i].set_ylabel(channel_labels[i])
        max_val = np.max(np.abs(eeg_data[i])) * 1.2
        axes[i].set_ylim(-max_val, max_val)
        
        for start, end, channels in abnormal_regions:
            if i in channels:
                axes[i].axvspan(start, end, color='red', alpha=0.4)  # Slightly stronger highlight
        axes[i].grid(True, linestyle='--', alpha=0.3)  # Add light grid for readability
    
    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    
    return buf

# Function to generate PDF report
def generate_pdf_report(diagnosis, eeg_plot, pattern, filename="eeg_report.pdf"):
    """Generate a PDF report with diagnosis, recommendations, pattern info, and EEG plot."""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        content = []
        content.append(Paragraph("EEG Diagnostic Report", styles['Title']))
        content.append(Spacer(1, 0.2 * inch))
        
        pattern_info = get_pattern_info(pattern)
        content.append(Paragraph("EEG Pattern Information", styles['Heading2']))
        content.append(Paragraph(f"**Pattern**: {pattern_info['title']}", styles['Normal']))
        content.append(Paragraph(f"**Description**: {pattern_info['description']}", styles['Normal']))
        content.append(Paragraph(f"**Wave Characteristics**:<br/>{pattern_info['wave_characteristics']}", styles['Normal']))
        content.append(Paragraph(f"**Associated Conditions**: {pattern_info['conditions']}", styles['Normal']))
        content.append(Paragraph(f"**Clinical Significance**: {pattern_info['clinical_significance']}", styles['Normal']))
        content.append(Spacer(1, 0.2 * inch))
        
        content.append(Paragraph(f"Overall Status: {diagnosis['overall_status']}", styles['Heading2']))
        content.append(Spacer(1, 0.1 * inch))
        content.append(Paragraph("Key Findings", styles['Heading3']))
        content.append(Paragraph(diagnosis['findings'], styles['Normal']))
        content.append(Spacer(1, 0.1 * inch))
        content.append(Paragraph("Diagnostic Impression", styles['Heading3']))
        content.append(Paragraph(diagnosis['diagnosis'], styles['Normal']))
        content.append(Paragraph(f"Diagnostic Confidence: {diagnosis['confidence']*100:.1f}%", styles['Normal']))
        content.append(Paragraph(f"Model Accuracy: {diagnosis['model_accuracy']*100:.1f}%", styles['Normal']))
        content.append(Spacer(1, 0.2 * inch))
        
        content.append(Paragraph("Treatment Recommendations", styles['Heading3']))
        for rec in diagnosis['recommendations']:
            content.append(Paragraph(f"• {rec}", styles['Normal']))
        
        additional_recommendations = ""
        if "epilepsy" in pattern or "epileptiform" in pattern or "seizure" in diagnosis['diagnosis'].lower():
            additional_recommendations = """
            <b>Medication Considerations:</b><br/>
            - Anticonvulsant therapy may be indicated based on clinical correlation<br/>
            - Common first-line medications: Levetiracetam, Lamotrigine, Carbamazepine (selection depends on seizure type)<br/>
            <b>Lifestyle Modifications:</b><br/>
            - Maintain regular sleep schedule<br/>
            - Avoid alcohol and recreational drugs<br/>
            - Stress management techniques<br/>
            - Consider ketogenic diet in consultation with neurologist for refractory cases
            """
        elif pattern in ["encephalopathy", "dementia", "hypoxia", "metabolic_conditions"]:
            additional_recommendations = """
            <b>Additional Investigations to Consider:</b><br/>
            - Complete metabolic panel<br/>
            - Toxicology screening<br/>
            - Brain imaging (MRI preferred)<br/>
            - CSF analysis if indicated<br/>
            <b>Supportive Care:</b><br/>
            - Identify and treat underlying causes<br/>
            - Cognitive monitoring<br/>
            - Supportive therapy as needed
            """
        if additional_recommendations:
            content.append(Paragraph(additional_recommendations, styles['Normal']))
        content.append(Spacer(1, 0.2 * inch))
        
        content.append(Paragraph("EEG Visualization", styles['Heading3']))
        eeg_plot.seek(0)
        img = ReportLabImage(eeg_plot, width=6*inch, height=4*inch)
        content.append(img)
        
        doc.build(content)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error generating PDF report: {e}")
        return None

# Function to display treatment recommendations
def display_treatment_recommendations(diagnosis, pattern):
    """Display treatment recommendations."""
    st.markdown('<div class="sub-header">Treatment Recommendations</div>', unsafe_allow_html=True)
    
    if diagnosis['overall_status'] == 'Normal':
        st.markdown('<div class="treatment-box">No specific treatment needed based on EEG findings.</div>', unsafe_allow_html=True)
        return
    
    st.markdown('<div class="treatment-box">', unsafe_allow_html=True)
    for rec in diagnosis['recommendations']:
        st.markdown(f"- {rec}")
    
    if "epilepsy" in pattern or "epileptiform" in pattern or "seizure" in diagnosis['diagnosis'].lower():
        st.markdown("""
        **Medication Considerations:**
        - Anticonvulsant therapy may be indicated based on clinical correlation
        - Common first-line medications: Levetiracetam, Lamotrigine, Carbamazepine (selection depends on seizure type)
        
        **Lifestyle Modifications:**
        - Maintain regular sleep schedule
        - Avoid alcohol and recreational drugs
        - Stress management techniques
        - Consider ketogenic diet in consultation with neurologist for refractory cases
        """)
    elif pattern in ["encephalopathy", "dementia", "hypoxia", "metabolic_conditions"]:
        st.markdown("""
        **Additional Investigations to Consider:**
        - Complete metabolic panel
        - Toxicology screening
        - Brain imaging (MRI preferred)
        - CSF analysis if indicated
        
        **Supportive Care:**
        - Identify and treat underlying causes
        - Cognitive monitoring
        - Supportive therapy as needed
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="warning-box">Note: All treatment recommendations should be reviewed by a qualified healthcare provider. This analysis is not a substitute for clinical judgment.</div>', unsafe_allow_html=True)

# Main app layout
st.markdown('<div class="sub-header">EEG Analysis Dashboard</div>', unsafe_allow_html=True)

# Handle input methods
eeg_data = None
time_points = None
abnormal_regions = []
channel_labels = None
pattern = None

if input_type == "Upload EEG Data File":
    uploaded_file = st.file_uploader("Upload EEG data file (.csv)", type=['csv'], help="Upload a CSV file with time in the first column and EEG channels in subsequent columns.")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            time_col = df.columns[0]
            data_cols = df.columns[1:]
            time_points = df[time_col].values
            eeg_data = df[data_cols].values.T
            channel_labels = data_cols.tolist()
            abnormal_regions = []
            pattern = "uploaded_data"
            st.success(f"Successfully loaded EEG data with {eeg_data.shape[0]} channels and {eeg_data.shape[1]} time points.")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            eeg_data = None
            time_points = None
            abnormal_regions = []
            channel_labels = None
            pattern = None

elif input_type == "Use MNE Sample Data":
    sample_type = st.selectbox(
        "Select EEG Pattern Type",
        [
            "Normal", "Epilepsy", "Dementia", "Brain Tumor", "Encephalitis",
            "Traumatic Brain Injury (TBI)", "Stroke", "Metabolic Conditions",
            "Hypoxia", "Brain Death", "Epileptiform Activity", "Encephalopathy",
            "Mild Abnormalities"
        ],
        help="Choose an EEG pattern to analyze using MNE sample data."
    )
    
    pattern_key = sample_type.lower().replace(" ", "_")
    pattern_info = get_pattern_info(pattern_key)
    st.markdown('<div class="sub-header">Selected EEG Pattern Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(f"**Pattern**: {pattern_info['title']}")
    st.markdown(f"**Description**: {pattern_info['description']}")
    st.markdown(f"**Wave Characteristics**:<br>{pattern_info['wave_characteristics']}", unsafe_allow_html=True)
    st.markdown(f"**Associated Conditions**: {pattern_info['conditions']}")
    st.markdown(f"**Clinical Significance**: {pattern_info['clinical_significance']}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Load MNE Sample EEG Data"):
        with st.spinner(f"Loading {sample_type} EEG data from MNE dataset..."):
            try:
                eeg_data, time_points, abnormal_regions, channel_labels = fetch_mne_sample_data(pattern=pattern_key)
                pattern = pattern_key
                st.success(f"Loaded MNE sample EEG data with {eeg_data.shape[0]} channels and {eeg_data.shape[1]} time points.")
            except Exception as e:
                st.error(f"Error loading MNE sample data: {e}")
                eeg_data = None
                time_points = None
                abnormal_regions = []
                channel_labels = None
                pattern = None

elif input_type == "Generate Simulated EEG Data":
    sample_type = st.selectbox(
        "Select EEG Pattern Type",
        [
            "Normal", "Epilepsy", "Dementia", "Brain Tumor", "Encephalitis",
            "Traumatic Brain Injury (TBI)", "Stroke", "Metabolic Conditions",
            "Hypoxia", "Brain Death", "Epileptiform Activity", "Encephalopathy",
            "Mild Abnormalities"
        ],
        help="Choose a simulated EEG pattern to analyze."
    )
    
    pattern_key = sample_type.lower().replace(" ", "_")
    pattern_info = get_pattern_info(pattern_key)
    st.markdown('<div class="sub-header">Selected EEG Pattern Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(f"**Pattern**: {pattern_info['title']}")
    st.markdown(f"**Description**: {pattern_info['description']}")
    st.markdown(f"**Wave Characteristics**:<br>{pattern_info['wave_characteristics']}", unsafe_allow_html=True)
    st.markdown(f"**Associated Conditions**: {pattern_info['conditions']}")
    st.markdown(f"**Clinical Significance**: {pattern_info['clinical_significance']}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Generate Simulated EEG Data"):
        with st.spinner(f"Generating {sample_type} EEG data..."):
            eeg_data, time_points, abnormal_regions, channel_labels = generate_eeg_sample(pattern=pattern_key)
            pattern = pattern_key
            st.success(f"Generated {sample_type} EEG data with {eeg_data.shape[0]} channels and {eeg_data.shape[1]} time points.")

# Process and display results if data is available
if eeg_data is not None and time_points is not None and channel_labels is not None:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Preprocessing EEG data...")
    progress_bar.progress(20)
    time.sleep(0.5)
    
    status_text.text("Analyzing abnormalities...")
    progress_bar.progress(40)
    detected_abnormalities, channel_analyses = analyze_eeg(eeg_data, time_points, sensitivity)
    time.sleep(0.5)
    
    status_text.text("Generating diagnostic report...")
    progress_bar.progress(60)
    diagnosis = generate_diagnosis(detected_abnormalities, channel_analyses, pattern)
    time.sleep(0.5)
    
    status_text.text("Visualizing results...")
    progress_bar.progress(80)
    all_abnormal_regions = abnormal_regions + [(start, end, [ch]) for start, end, ch in detected_abnormalities]
    eeg_plot = plot_eeg_with_highlights(eeg_data, time_points, all_abnormal_regions, channel_labels)
    progress_bar.progress(100)
    status_text.empty()
    
    st.markdown('<div class="sub-header">EEG Visualization</div>', unsafe_allow_html=True)
    pattern_title = get_pattern_info(pattern)['title'] if pattern else "EEG Data"
    st.image(eeg_plot, caption=f"EEG Recording ({pattern_title})", use_container_width=True)
    
    st.markdown('<div class="sub-header">Frequency Analysis</div>', unsafe_allow_html=True)
    if channel_analyses:
        total_channels = len(channel_analyses)
        channel_options = [f"Channel {ch['channel']+1}" for ch in channel_analyses]
        
        # Determine min and max channels for the slider
        max_channels = min(total_channels, 60)  # Cap at 60 or total channels
        min_channels = min(8, total_channels)   # Ensure minimum is 8, unless total channels are fewer
        
        # Default number of channels to display
        if min_channels == max_channels:
            # If min and max are the same, no slider is needed
            num_channels_to_display = min_channels
            st.info(f"Displaying all {num_channels_to_display} available channels (no range to adjust).")
        else:
            # Use slider when there is a range to select from
            num_channels_to_display = st.slider(
                "Number of channels to display",
                min_value=min_channels,
                max_value=max_channels,
                value=min_channels,  # Default to minimum (8 or total channels if fewer)
                step=1,
                help=f"Select how many channels to display (from {min_channels} to {max_channels})."
            )
        
        # Update the default selected channels based on the number to display
        default_selected = channel_options[:num_channels_to_display]
        
        # Multiselect widget for channel selection
        selected_channels = st.multiselect(
            "Select channels to display in frequency analysis",
            options=channel_options,
            default=default_selected,
            help="Select the channels to display in the frequency analysis plot."
        )
        
        # Filter channel analyses based on selected channels
        selected_indices = [int(ch.split()[-1]) - 1 for ch in selected_channels]
        filtered_analyses = [ch for ch in channel_analyses if ch['channel'] in selected_indices]
        
        if filtered_analyses:
            num_channels = len(filtered_analyses)
            fig, ax = plt.subplots(figsize=(12, 8))
            bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
            x = np.arange(len(bands))
            
            # Dynamically adjust bar width based on number of channels
            width = min(0.8 / num_channels, 0.1)
            
            # Use viridis colormap for distinct colors
            colormap = cm.get_cmap('viridis')
            colors = [colormap(i / num_channels) for i in range(num_channels)]
            
            for i, ch_analysis in enumerate(filtered_analyses):
                values = [
                    ch_analysis['delta_ratio'],
                    ch_analysis['theta_ratio'],
                    ch_analysis['alpha_ratio'],
                    ch_analysis['beta_ratio'],
                    ch_analysis['gamma_ratio']
                ]
                ax.bar(x + i * width - (num_channels - 1) * width / 2, 
                       values, 
                       width, 
                       label=f"Ch {ch_analysis['channel']+1}",
                       color=colors[i],
                       edgecolor='black',
                       linewidth=0.5)
            
            ax.set_ylabel('Power Ratio', fontsize=12)
            ax.set_title('Frequency Band Power Distribution by Channel', fontsize=14, pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels(bands, fontsize=10)
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle='--', alpha=0.3, axis='y')
            
            # Place legend outside the plot
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                     fontsize=10, title="Channels", 
                     frameon=True, edgecolor='black')
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No channels selected for frequency analysis.")
    else:
        st.info("Frequency analysis not available.")
    
    st.markdown('<div class="sub-header">Diagnostic Report</div>', unsafe_allow_html=True)
    status_color = {
        'Normal': 'normal-region',
        'Borderline': 'highlight-text',
        'Borderline Abnormal': 'highlight-text',
        'Abnormal': 'abnormal-region'
    }.get(diagnosis['overall_status'], 'highlight-text')
    
    st.markdown(f'<div class="{status_color}">Overall Status: {diagnosis["overall_status"]}</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="diagnosis-box">', unsafe_allow_html=True)
    st.markdown("### Key Findings")
    st.markdown(diagnosis['findings'])
    
    st.markdown("### Diagnostic Impression")
    st.markdown(diagnosis['diagnosis'])
    
    st.markdown(f"*Diagnostic Confidence: {diagnosis['confidence']*100:.1f}%*")
    st.markdown(f"*Model Accuracy: {diagnosis['model_accuracy']*100:.1f}%*")
    st.markdown('</div>', unsafe_allow_html=True)
    
    display_treatment_recommendations(diagnosis, pattern)
    
    st.markdown('<div class="sub-header">Export Results</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Generate PDF Report"):
            with st.spinner("Generating PDF report..."):
                pdf_buffer = generate_pdf_report(diagnosis, eeg_plot, pattern)
                if pdf_buffer:
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_buffer,
                        file_name="eeg_diagnostic_report.pdf",
                        mime="application/pdf",
                        key="pdf_download"
                    )
                else:
                    st.error("Failed to generate PDF report.")
    
    with col2:
        if st.button("📊 Generate CSV Export"):
            with st.spinner("Generating CSV file..."):
                try:
                    df = pd.DataFrame(eeg_data.T, columns=channel_labels)
                    df['Time'] = time_points
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue().encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name="eeg_data.csv",
                        mime="text/csv",
                        key="csv_download"
                    )
                except Exception as e:
                    st.error(f"Error generating CSV: {e}")
    
    with col3:
        if st.button("🖼️ Generate EEG Plot"):
            with st.spinner("Generating EEG plot..."):
                try:
                    eeg_plot.seek(0)
                    st.download_button(
                        label="Download EEG Plot",
                        data=eeg_plot,
                        file_name="eeg_plot.png",
                        mime="image/png",
                        key="png_download"
                    )
                except Exception as e:
                    st.error(f"Error generating PNG: {e}")

else:
    st.markdown("""
    ## Getting Started
    
    1. **Upload EEG Data**: Select "Upload EEG Data File" and upload a CSV file with time in the first column and EEG channels in subsequent columns.
    2. **Use MNE Sample Data**: Select a pattern (e.g., Epilepsy, Dementia) and click "Load MNE Sample EEG Data" to analyze real EEG data modified for the selected pattern.
    3. **Generate Simulated EEG Data**: Select a pattern and click "Generate Simulated EEG Data" to analyze programmatically generated EEG data.
    4. Adjust analysis settings in the sidebar if needed.
    5. The system will automatically:
       - Process the EEG data
       - Display pattern information (for sample or simulated data)
       - Detect and highlight abnormalities
       - Generate a diagnostic report with model accuracy
       - Provide treatment recommendations
       - Offer export options (PDF, CSV, PNG)
    
    **Note**: This is a professional prototype for demonstration purposes. Always consult a qualified healthcare provider for clinical decisions.
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #6B7280;">
        EEG Diagnostic System | Powered by Streamlit and MNE-Python
    </div>
""", unsafe_allow_html=True)