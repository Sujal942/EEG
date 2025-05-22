import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import time

# Set page configuration
st.set_page_config(
    page_title="EEG Report Analyzer",
    page_icon="🧠",
    layout="wide",
)

# Custom CSS for styling
st.markdown("""
    <style>
        .main-header {
            font-size: 36px;
            font-weight: bold;
            color: #2E86C1;
            margin-bottom: 20px;
        }
        .sub-header {
            font-size: 24px;
            font-weight: bold;
            color: #2874A6;
            margin-top: 30px;
            margin-bottom: 10px;
        }
        .highlight-text {
            background-color: #F9E79F;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0px;
        }
        .abnormal-region {
            background-color: rgba(231, 76, 60, 0.3);
            border-radius: 5px;
            padding: 5px;
        }
        .normal-region {
            background-color: rgba(46, 204, 113, 0.3);
            border-radius: 5px;
            padding: 5px;
        }
        .diagnosis-box {
            background-color: #EBF5FB;
            border-left: 5px solid #3498DB;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0px;
        }
        .treatment-box {
            background-color: #E8F8F5;
            border-left: 5px solid #1ABC9C;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0px;
        }
        .warning-box {
            background-color: #FDEDEC;
            border-left: 5px solid #E74C3C;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0px;
        }
    </style>
    <div class="main-header">EEG Report Analysis and Prediction System</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/frontend/src/assets/branding/streamlit-mark-color.svg", width=100)
    st.markdown("## Input Options")
    
    input_type = st.radio(
        "Select Input Method",
        ["Upload EEG Data File", "Use Sample Data", "Manual Parameter Input"]
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
    This application analyzes EEG data to identify potential abnormalities 
    and provide diagnostic insights with treatment recommendations.
    
    **Note:** This is a prototype and should not replace professional medical advice.
    """)

# Main function to generate EEG sample data
def generate_eeg_sample(duration=10, sampling_rate=250, channels=8, abnormal=True):
    """Generate sample EEG data with optional abnormalities"""
    time_points = np.arange(0, duration, 1/sampling_rate)
    n_samples = len(time_points)
    
    # Base frequencies for different brain waves
    delta = np.sin(2 * np.pi * 2 * time_points)  # 2 Hz (delta wave)
    theta = np.sin(2 * np.pi * 6 * time_points)  # 6 Hz (theta wave)
    alpha = np.sin(2 * np.pi * 10 * time_points)  # 10 Hz (alpha wave)
    beta = np.sin(2 * np.pi * 20 * time_points)  # 20 Hz (beta wave)
    
    # Generate clean EEG data
    eeg_data = np.zeros((channels, n_samples))
    
    for i in range(channels):
        # Mix different wave components with random weights
        a, b, c, d = np.random.rand(4)
        eeg_data[i] = (a * delta + b * theta + c * alpha + d * beta) / (a + b + c + d)
        # Add some random noise
        eeg_data[i] += np.random.normal(0, 0.1, n_samples)
    
    # Add abnormalities if requested (spike-wave complexes typical in some seizures)
    abnormal_regions = []
    if abnormal:
        # Add 2-3 abnormal regions
        num_abnormalities = np.random.randint(2, 4)
        for _ in range(num_abnormalities):
            # Random start time for abnormality
            abnormal_start = np.random.randint(0, n_samples - sampling_rate)
            abnormal_duration = np.random.randint(sampling_rate // 2, sampling_rate * 2)
            abnormal_end = min(abnormal_start + abnormal_duration, n_samples)
            
            # Select random channels for the abnormality
            affected_channels = np.random.choice(channels, size=np.random.randint(2, channels+1), replace=False)
            
            # Create spike-wave pattern
            for ch in affected_channels:
                # Create a spike-wave complex
                spike_freq = 3  # Hz (typical for absence seizures)
                num_cycles = int((abnormal_end - abnormal_start) / sampling_rate * spike_freq)
                
                for c in range(num_cycles):
                    cycle_start = abnormal_start + int(c * sampling_rate / spike_freq)
                    spike_pos = cycle_start + int(sampling_rate / (spike_freq * 4))
                    
                    if spike_pos < abnormal_end:
                        # Add a spike (sharp peak)
                        amplitude = np.random.uniform(1.5, 2.5)
                        width = int(sampling_rate / 20)  # 50ms spike
                        for j in range(max(0, spike_pos-width), min(spike_pos+width, n_samples)):
                            dist = abs(j - spike_pos)
                            if dist < width:
                                spike_val = amplitude * (1 - dist/width)
                                eeg_data[ch, j] += spike_val
                
                # Add high frequency oscillations
                hfo = np.sin(2 * np.pi * 80 * time_points[abnormal_start:abnormal_end]) * 0.3
                eeg_data[ch, abnormal_start:abnormal_end] += hfo
            
            abnormal_regions.append((abnormal_start/sampling_rate, abnormal_end/sampling_rate, affected_channels))
    
    return eeg_data, time_points, abnormal_regions

# Function to analyze EEG data and detect abnormalities
def analyze_eeg(eeg_data, time_points, sensitivity=0.7):
    """Analyze EEG data to detect abnormalities"""
    num_channels, num_samples = eeg_data.shape
    sampling_rate = int(num_samples / time_points[-1])
    
    detected_abnormalities = []
    channel_analyses = []
    
    # Analyze each channel
    for ch in range(num_channels):
        channel_data = eeg_data[ch]
        
        # Calculate basic statistics
        mean_val = np.mean(channel_data)
        std_val = np.std(channel_data)
        
        # Detect potential abnormalities based on amplitude thresholds
        threshold = 2.0 + (1.0 - sensitivity) * 2.0  # Adjustable threshold based on sensitivity
        abnormal_indices = np.where(np.abs(channel_data - mean_val) > threshold * std_val)[0]
        
        # Group consecutive abnormal points into regions
        if len(abnormal_indices) > 0:
            abnormal_regions = []
            region_start = abnormal_indices[0]
            for i in range(1, len(abnormal_indices)):
                if abnormal_indices[i] - abnormal_indices[i-1] > sampling_rate/5:  # If gap is more than 200ms
                    region_end = abnormal_indices[i-1]
                    if (region_end - region_start) > sampling_rate/10:  # If region is more than 100ms
                        abnormal_regions.append((region_start/sampling_rate, region_end/sampling_rate))
                    region_start = abnormal_indices[i]
            
            # Add the last region
            if len(abnormal_indices) > 0:
                region_end = abnormal_indices[-1]
                if (region_end - region_start) > sampling_rate/10:
                    abnormal_regions.append((region_start/sampling_rate, region_end/sampling_rate))
            
            for start, end in abnormal_regions:
                detected_abnormalities.append((start, end, ch))
        
        # Calculate frequency domain features
        if num_samples > 0:
            try:
                # Use FFT to get frequency components
                fft_vals = np.abs(np.fft.rfft(channel_data))
                fft_freq = np.fft.rfftfreq(num_samples, 1.0/sampling_rate)
                
                # Calculate power in different frequency bands
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

# Function to generate diagnostic report based on EEG analysis
def generate_diagnosis(abnormalities, channel_analyses):
    """Generate diagnostic information based on EEG abnormalities"""
    
    if not abnormalities and not channel_analyses:
        return {
            'overall_status': 'Normal',
            'findings': 'No significant abnormalities detected in the EEG recording.',
            'diagnosis': 'Normal EEG within physiological limits.',
            'confidence': 0.95,
            'recommendations': [
                'No specific EEG-based interventions required.',
                'Continue monitoring if symptoms persist.'
            ]
        }
    
    # Count abnormalities
    num_abnormalities = len(abnormalities)
    
    # Analyze frequency patterns
    avg_delta = np.mean([ch['delta_ratio'] for ch in channel_analyses]) if channel_analyses else 0
    avg_theta = np.mean([ch['theta_ratio'] for ch in channel_analyses]) if channel_analyses else 0
    avg_alpha = np.mean([ch['alpha_ratio'] for ch in channel_analyses]) if channel_analyses else 0
    avg_beta = np.mean([ch['beta_ratio'] for ch in channel_analyses]) if channel_analyses else 0
    avg_gamma = np.mean([ch['gamma_ratio'] for ch in channel_analyses]) if channel_analyses else 0
    
    # Determine diagnosis based on patterns
    if num_abnormalities > 5 and avg_delta > 0.4:
        diagnosis = {
            'overall_status': 'Abnormal',
            'findings': 'Multiple high-amplitude sharp waves and spike-wave complexes detected with elevated delta activity.',
            'diagnosis': 'Findings consistent with epileptiform activity, suggestive of seizure disorder.',
            'confidence': 0.85,
            'recommendations': [
                'Neurology consultation recommended',
                'Consider anticonvulsant therapy evaluation',
                'Follow-up EEG in 3-6 months',
                'Avoid sleep deprivation and other seizure triggers'
            ]
        }
    elif num_abnormalities > 3 and avg_theta > 0.3:
        diagnosis = {
            'overall_status': 'Abnormal',
            'findings': 'Intermittent sharp waves and slow wave activity with elevated theta rhythm.',
            'diagnosis': 'Findings suggestive of focal encephalopathy or mild seizure activity.',
            'confidence': 0.75,
            'recommendations': [
                'Neurology consultation recommended',
                'Consider brain imaging (MRI)',
                'Follow-up EEG in 3 months',
                'Monitor for clinical symptoms'
            ]
        }
    elif avg_beta > 0.5:
        diagnosis = {
            'overall_status': 'Borderline Abnormal',
            'findings': 'Excessive beta activity detected throughout the recording.',
            'diagnosis': 'Findings may indicate anxiety, medication effect (benzodiazepines), or mild cortical irritability.',
            'confidence': 0.7,
            'recommendations': [
                'Review current medications',
                'Consider anxiety assessment',
                'Follow-up as clinically indicated'
            ]
        }
    elif avg_alpha < 0.1:
        diagnosis = {
            'overall_status': 'Borderline Abnormal',
            'findings': 'Reduced alpha rhythm with poor organization.',
            'diagnosis': 'Findings suggestive of encephalopathy or altered mental status.',
            'confidence': 0.65,
            'recommendations': [
                'Clinical correlation with mental status',
                'Consider metabolic and toxic causes',
                'Follow-up EEG if symptoms persist'
            ]
        }
    else:
        diagnosis = {
            'overall_status': 'Borderline',
            'findings': 'Minor EEG abnormalities detected but no definitive pathological pattern.',
            'diagnosis': 'Nonspecific EEG changes of uncertain clinical significance.',
            'confidence': 0.6,
            'recommendations': [
                'Clinical correlation recommended',
                'Consider follow-up EEG if symptoms persist',
                'Monitor for development of clearer patterns'
            ]
        }
    
    return diagnosis

# Function to plot EEG data with highlighted abnormalities
def plot_eeg_with_highlights(eeg_data, time_points, abnormal_regions, channel_labels=None):
    """Plot EEG data with highlighted abnormal regions"""
    num_channels = eeg_data.shape[0]
    
    # Create default channel labels if not provided
    if channel_labels is None:
        channel_labels = [f"Channel {i+1}" for i in range(num_channels)]
    
    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 2*num_channels), sharex=True)
    
    # Handle case of single channel
    if num_channels == 1:
        axes = [axes]
    
    # Plot each channel
    for i in range(num_channels):
        axes[i].plot(time_points, eeg_data[i], 'k-', linewidth=1)
        axes[i].set_ylabel(channel_labels[i])
        
        # Set y-limits with some padding
        max_val = np.max(np.abs(eeg_data[i])) * 1.2
        axes[i].set_ylim(-max_val, max_val)
        
        # Highlight abnormal regions
        for start, end, channels in abnormal_regions:
            if i in channels:
                axes[i].axvspan(start, end, color='red', alpha=0.3)
    
    # Add labels
    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    
    return buf

# Function to display treatment recommendations
def display_treatment_recommendations(diagnosis):
    """Display treatment recommendations based on diagnosis"""
    
    st.markdown('<div class="sub-header">Treatment Recommendations</div>', unsafe_allow_html=True)
    
    if diagnosis['overall_status'] == 'Normal':
        st.markdown('<div class="treatment-box">No specific treatment needed based on EEG findings.</div>', unsafe_allow_html=True)
        return
    
    st.markdown('<div class="treatment-box">', unsafe_allow_html=True)
    
    for rec in diagnosis['recommendations']:
        st.markdown(f"- {rec}")
    
    # Additional information based on diagnosis
    if "epileptiform" in diagnosis['diagnosis'].lower() or "seizure" in diagnosis['diagnosis'].lower():
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
    elif "encephalopathy" in diagnosis['diagnosis'].lower():
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
st.markdown('<div class="sub-header">EEG Analysis Tool</div>', unsafe_allow_html=True)

# Handle different input methods
if input_type == "Upload EEG Data File":
    uploaded_file = st.file_uploader("Upload EEG data file (.csv, .edf, .txt)", type=['csv', 'txt', 'edf'])
    
    if uploaded_file is not None:
        try:
            # For this demo, we'll assume CSV format with time in first column and channel data in subsequent columns
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                time_col = df.columns[0]
                data_cols = df.columns[1:]
                
                time_points = df[time_col].values
                eeg_data = df[data_cols].values.T  # Transpose to get channels as first dimension
                
                st.success(f"Successfully loaded data with {eeg_data.shape[0]} channels and {eeg_data.shape[1]} time points")
                
                # For demo purposes, generate synthetic abnormalities
                # In a real app, you would analyze the actual data
                abnormal_regions = []
                
            else:
                st.error("This demo currently only supports CSV files. For other formats, please use the sample data option.")
                eeg_data = None
                time_points = None
                abnormal_regions = []
        except Exception as e:
            st.error(f"Error loading file: {e}")
            eeg_data = None
            time_points = None
            abnormal_regions = []
    else:
        eeg_data = None
        time_points = None
        abnormal_regions = []

elif input_type == "Use Sample Data":
    sample_type = st.selectbox(
        "Select Sample Type",
        ["Normal EEG", "Mild Abnormalities", "Epileptiform Activity", "Encephalopathy Pattern"]
    )
    
    if st.button("Generate Sample Data"):
        with st.spinner("Generating sample EEG data..."):
            # Generate different types of sample data based on selection
            if sample_type == "Normal EEG":
                eeg_data, time_points, abnormal_regions = generate_eeg_sample(duration=10, channels=8, abnormal=False)
            elif sample_type == "Mild Abnormalities":
                eeg_data, time_points, abnormal_regions = generate_eeg_sample(duration=10, channels=8, abnormal=True)
                # Modify some of the abnormalities to be milder
                abnormal_regions = abnormal_regions[:1]  # Limit to fewer abnormalities
            elif sample_type == "Epileptiform Activity":
                eeg_data, time_points, _ = generate_eeg_sample(duration=10, channels=8, abnormal=True)
                # Create more pronounced spike-wave patterns
                abnormal_regions = []
                for i in range(3):
                    start = 2 + i * 2.5
                    end = start + 1.5
                    channels = np.random.choice(8, size=5, replace=False)
                    abnormal_regions.append((start, end, channels))
                    
                    # Add spike-wave complexes
                    for ch in channels:
                        for t in np.arange(start, end, 0.3):
                            idx = int(t * 250)  # Assuming 250 Hz sampling rate
                            if idx < len(time_points) - 10:
                                eeg_data[ch, idx:idx+5] += 2.0  # Add spike
                                eeg_data[ch, idx+5:idx+10] -= 1.0  # Add wave
            else:  # Encephalopathy Pattern
                eeg_data, time_points, _ = generate_eeg_sample(duration=10, channels=8, abnormal=False)
                # Create slow-wave pattern typical of encephalopathy
                abnormal_regions = [(2, 8, np.arange(8))]
                
                slow_wave = np.sin(2 * np.pi * 1.5 * time_points) * 0.8  # 1.5 Hz slow wave
                for ch in range(8):
                    idx_start = int(2 * 250)
                    idx_end = int(8 * 250)
                    if idx_end <= len(time_points):
                        eeg_data[ch, idx_start:idx_end] += slow_wave[idx_start:idx_end]
            
            st.success("Sample data generated successfully!")
    else:
        eeg_data = None
        time_points = None
        abnormal_regions = []

else:  # Manual Parameter Input
    st.markdown("Please specify parameters to generate synthetic EEG data:")
    
    duration = st.slider("Recording Duration (seconds)", 5, 30, 10)
    num_channels = st.slider("Number of Channels", 1, 16, 8)
    has_abnormalities = st.checkbox("Include Abnormalities", value=True)
    
    if st.button("Generate Data"):
        with st.spinner("Generating EEG data with specified parameters..."):
            eeg_data, time_points, abnormal_regions = generate_eeg_sample(
                duration=duration, 
                channels=num_channels, 
                abnormal=has_abnormalities
            )
            st.success("Data generated successfully!")
    else:
        eeg_data = None
        time_points = None
        abnormal_regions = []

# Process and analyze the data if available
if eeg_data is not None and time_points is not None:
    # Show progress bar for analysis
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Analysis steps with progress updates
    status_text.text("Preprocessing EEG data...")
    progress_bar.progress(20)
    time.sleep(0.5)  # Simulating processing time
    
    status_text.text("Detecting abnormalities...")
    progress_bar.progress(40)
    detected_abnormalities, channel_analyses = analyze_eeg(eeg_data, time_points, sensitivity)
    time.sleep(0.5)
    
    status_text.text("Generating diagnostic report...")
    progress_bar.progress(60)
    diagnosis = generate_diagnosis(detected_abnormalities, channel_analyses)
    time.sleep(0.5)
    
    status_text.text("Visualizing results...")
    progress_bar.progress(80)
    
    # Combine detected and simulated abnormalities for visualization
    # In a real application, you would only use detected abnormalities
    all_abnormal_regions = abnormal_regions + [(start, end, [ch]) for start, end, ch in detected_abnormalities]
    
    eeg_plot = plot_eeg_with_highlights(eeg_data, time_points, all_abnormal_regions)
    progress_bar.progress(100)
    status_text.empty()
    
    # Display results
    st.markdown('<div class="sub-header">EEG Visualization with Highlighted Abnormalities</div>', unsafe_allow_html=True)
    st.image(eeg_plot, caption="EEG Recording with Highlighted Abnormal Regions", use_column_width=True)
    
    # Show Frequency Analysis
    st.markdown('<div class="sub-header">Frequency Analysis</div>', unsafe_allow_html=True)
    
    if channel_analyses:
        # Create frequency analysis chart
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        x = np.arange(len(bands))
        width = 0.1
        
        for i, ch_analysis in enumerate(channel_analyses):
            values = [
                ch_analysis['delta_ratio'], 
                ch_analysis['theta_ratio'],
                ch_analysis['alpha_ratio'],
                ch_analysis['beta_ratio'],
                ch_analysis['gamma_ratio']
            ]
            ax.bar(x + i*width - (len(channel_analyses)-1)*width/2, values, width, label=f"Ch {ch_analysis['channel']+1}")
        
        ax.set_ylabel('Power Ratio')
        ax.set_title('Frequency Band Power Distribution by Channel')
        ax.set_xticks(x)
        ax.set_xticklabels(bands)
        ax.legend(loc='best', ncol=min(8, len(channel_analyses)))
        
        st.pyplot(fig)
    else:
        st.info("Frequency analysis not available.")
    
    # Display diagnostic report
    st.markdown('<div class="sub-header">Diagnostic Report</div>', unsafe_allow_html=True)
    
    # Status badge
    status_color = {
        'Normal': 'normal-region',
        'Borderline': 'highlight-text',
        'Borderline Abnormal': 'highlight-text',
        'Abnormal': 'abnormal-region'
    }.get(diagnosis['overall_status'], 'highlight-text')
    
    st.markdown(f'<div class="{status_color}">Overall Status: {diagnosis["overall_status"]}</div>', unsafe_allow_html=True)
    
    # Diagnostic findings
    st.markdown('<div class="diagnosis-box">', unsafe_allow_html=True)
    st.markdown("### Key Findings")
    st.markdown(diagnosis['findings'])
    
    st.markdown("### Diagnostic Impression")
    st.markdown(diagnosis['diagnosis'])
    
    st.markdown(f"*Diagnostic confidence: {diagnosis['confidence']*100:.1f}%*")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Treatment recommendations
    display_treatment_recommendations(diagnosis)
    
    # Export options
    st.markdown('<div class="sub-header">Export Options</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export Report (PDF)"):
            st.info("PDF export functionality would be implemented here in a full application.")
    
    with col2:
        if st.button("📊 Export Data (CSV)"):
            st.info("CSV export functionality would be implemented here.")
    
    with col3:
        if st.button("🖼️ Export Images"):
            st.info("Image export functionality would be implemented here.")

else:
    # Show instructions if no data is loaded
    st.markdown("""
    ## How to Use This Tool
    
    1. Select an input method from the sidebar
    2. Either upload your EEG data file or generate sample data
    3. Adjust analysis settings in the sidebar if needed
    4. The system will:
       - Process the EEG data
       - Highlight abnormal regions
       - Provide a diagnostic report
       - Suggest appropriate treatments
    
    **Note:** This is a demonstration prototype. In a clinical setting, this tool would be integrated with professional EEG equipment and validated algorithms.
    """)
    
    # Show example images
    st.markdown("### Example EEG Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Normal EEG**")
        # Generate a normal EEG sample for display
        normal_eeg, normal_time, _ = generate_eeg_sample(duration=5, channels=4, abnormal=False)
        normal_plot = plot_eeg_with_highlights(normal_eeg, normal_time, [])
        st.image(normal_plot, use_column_width=True)
    
    with col2:
        st.markdown("**Abnormal EEG (Epileptiform Activity)**")
        # Generate an abnormal EEG sample for display
        abnormal_eeg, abnormal_time, abnormal_regions = generate_eeg_sample(duration=5, channels=4, abnormal=True)
        abnormal_plot = plot_eeg_with_highlights(abnormal_eeg, abnormal_time, abnormal_regions)
        st.image(abnormal_plot, use_column_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666;">
        EEG Report Analysis and Prediction System
    </div>
""", unsafe_allow_html=True)

# Additional functionality that could be implemented in a full version:
# 1. User authentication system for medical professionals