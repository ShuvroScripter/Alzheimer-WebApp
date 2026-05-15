import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torchvision.transforms as transforms
from captum.attr import IntegratedGradients, Saliency
import os
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Set page configuration
st.set_page_config(
    page_title="Alzheimer's Detection - Clinical AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Medical CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .diagnosis-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #1f77b4;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .upload-container {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        background-color: #f8f9fa;
        margin-bottom: 20px;
    }
    .result-section {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .compact-image {
        max-width: 280px;
        max-height: 280px;
        margin: 0 auto;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border: 2px solid #e9ecef;
    }
    .analysis-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .image-container {
        text-align: center;
        padding: 15px;
        background: white;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ----------- Device Setup -----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- CORRECTED CLASS MAPPING -----------
# Based on your folder structure: Final AD JPEG, Final CN JPEG, Final MCI JPEG
categories = ["Alzheimer's Disease (AD)", "Cognitively Normal (CN)", "Mild Cognitive Impairment (MCI)"]
NUM_CLASSES = len(categories)

# ----------- Dataset Paths -----------
train_dir = r"C:\Users\Shuv\Alzheimers-ADNI\train"
test_dir  = r"C:\Users\Shuv\Alzheimers-ADNI\test"

# ----------- MODEL ARCHITECTURE (From your training code) -----------
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(F.relu(g1 + x1))
        return x * psi

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ModifiedSwin(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedSwin, self).__init__()
        self.base = timm.create_model(
            'swin_tiny_patch4_window7_224', pretrained=True,
            num_classes=0, global_pool=''
        )
        self.positional_encoding = PositionalEncoding(768, 5000)
        self.att_gate = AttentionGate(768, 768, 384)
        self.extra_transformer = nn.TransformerEncoderLayer(
            d_model=768, nhead=8, dim_feedforward=2048, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.extra_transformer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.base.forward_features(x)
        features = self.positional_encoding(features)
        features_2d = features.permute(0, 3, 1, 2)
        att_out = self.att_gate(features_2d, features_2d)
        enhanced = self.transformer(att_out.flatten(2).transpose(1, 2))
        out = self.classifier(enhanced.mean(dim=1))
        return out

# ----------- CORRECTED DATASET CLASS -----------
class CorrectedAlzheimerDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        
        # CORRECTED MAPPING based on your folder structure
        self.folder_to_class = {
            'Final AD JPEG': 0,   # Alzheimer's Disease → Class 0
            'Final CN JPEG': 1,   # Cognitively Normal → Class 1  
            'Final MCI JPEG': 2   # Mild Cognitive Impairment → Class 2
        }
        
        # Rebuild samples with correct mapping
        corrected_samples = []
        for path, original_class_idx in self.samples:
            folder_name = os.path.basename(os.path.dirname(path))
            if folder_name in self.folder_to_class:
                corrected_class_idx = self.folder_to_class[folder_name]
                corrected_samples.append((path, corrected_class_idx))
            else:
                corrected_samples.append((path, original_class_idx))
        
        self.samples = corrected_samples
        self.targets = [s[1] for s in corrected_samples]
        
        # Update class_to_idx to reflect correct mapping
        self.class_to_idx = {
            "Alzheimer's Disease (AD)": 0,
            "Cognitively Normal (CN)": 1, 
            "Mild Cognitive Impairment (MCI)": 2
        }
        self.classes = list(self.class_to_idx.keys())

# ----------- XAI Class -----------
class ClinicalXAI:
    def __init__(self, model, device, categories):
        self.model = model
        self.device = device
        self.categories = categories
        self.model.eval()
        self.ig = IntegratedGradients(model)
    
    def denormalize(self, tensor):
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(self.device)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(self.device)
        return tensor * std + mean
    
    def generate_explanation(self, image_tensor, predicted_label):
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = F.softmax(output, dim=1)[0]
            confidence = probs[predicted_label].item()
        
        # Generate integrated gradients
        target = torch.tensor([predicted_label], device=self.device)
        ig_attr = self.ig.attribute(image_tensor, target=target, n_steps=50)
        
        original_img = self.denormalize(image_tensor.squeeze(0)).cpu().permute(1, 2, 0).numpy()
        original_img = np.clip(original_img, 0, 1)
        
        ig_heatmap = ig_attr.squeeze(0).cpu().detach().numpy()
        ig_heatmap = np.sum(np.abs(ig_heatmap), axis=0)
        if ig_heatmap.max() > 0:
            ig_heatmap = (ig_heatmap - ig_heatmap.min()) / (ig_heatmap.max() - ig_heatmap.min())
        
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * ig_heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        heatmap_colored = heatmap_colored / 255.0
        
        return original_img, ig_heatmap, heatmap_colored, probs.cpu().numpy(), confidence

# ----------- Load Model -----------
@st.cache_resource
def load_model():
    model = ModifiedSwin(NUM_CLASSES).to(device)
    
    try:
        model_paths = [
            'modified_swin_alzheimer_model_final_103.pth',
            'modified_swin_alzheimer_model.pth'
        ]
        
        model_loaded = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
                st.sidebar.success(f"✅ Model loaded from {model_path}")
                model_loaded = True
                break
        
        if not model_loaded:
            st.sidebar.warning("❌ No pre-trained model found. Using initialized weights.")
            
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)[:100]}...")
    
    model.eval()
    return model

# ----------- Image Preprocessing -----------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0).to(device)

# ----------- Clinical Recommendations -----------
def get_clinical_recommendation(predicted_label, confidence):
    recommendations = []
    
    if predicted_label == 0:  # AD
        recommendations.append("**Immediate Actions:**")
        recommendations.append("• Refer to neurologist for comprehensive evaluation")
        recommendations.append("• Schedule cognitive assessment (MMSE/MoCA)")
        recommendations.append("• Consider CSF biomarkers or amyloid PET imaging")
        
    elif predicted_label == 1:  # CN
        recommendations.append("**Maintenance Actions:**")
        recommendations.append("• Continue regular health screenings")
        recommendations.append("• Maintain cognitive and physical activity")
        recommendations.append("• Monitor for any cognitive changes")
        
    else:  # MCI
        recommendations.append("**Recommended Actions:**")
        recommendations.append("• Annual cognitive monitoring")
        recommendations.append("• Lifestyle interventions and cognitive training")
        recommendations.append("• Cardiovascular risk factor management")
    
    if confidence < 0.7:
        recommendations.append("\n**Note:** Lower confidence - clinical correlation recommended")
    
    return recommendations

# ----------- Upload & Analyze Page -----------
def upload_analyze_page():
    st.markdown('<div class="main-header">🏥 Alzheimer\'s Disease Detection</div>', unsafe_allow_html=True)
    st.markdown("### Clinical AI Analysis System")
    
    # Display class mapping info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Class Mapping")
    st.sidebar.write("**Final AD JPEG** → Alzheimer's Disease")
    st.sidebar.write("**Final CN JPEG** → Cognitively Normal") 
    st.sidebar.write("**Final MCI JPEG** → Mild Cognitive Impairment")
    
    # File upload section
    st.markdown("---")
    st.subheader("Upload MRI Scan")
    
    uploaded_file = st.file_uploader(
        "Choose a brain MRI image", 
        type=['jpg', 'jpeg', 'png', 'tiff', 'bmp'],
        help="Upload T1-weighted MRI scan for analysis"
    )
    
    if uploaded_file is not None:
        # Create two columns: left for image, right for analysis
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            # Display uploaded image with controlled compact size
            image = Image.open(uploaded_file)
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.markdown('<div class="compact-image">', unsafe_allow_html=True)
            st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # File info
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size // 1024} KB")
            st.write(f"**Format:** {uploaded_file.type}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Analyze button below the image
            if st.button("🧠 Analyze MRI Scan", type="primary", use_container_width=True):
                st.session_state.analyze_clicked = True
        
        with col2:
            if st.session_state.get('analyze_clicked', False):
                with st.spinner("🔬 AI Analysis in Progress..."):
                    analyze_image(image, uploaded_file.name)

def analyze_image(image, filename):
    # Load model and perform analysis
    model = load_model()
    xai = ClinicalXAI(model, device, categories)
    
    image_tensor = preprocess_image(image)
    
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)[0]
        predicted_label = output.argmax(1).item()
        confidence = probs[predicted_label].item()
    
    # Generate XAI explanations
    original_img, heatmap, colored_heatmap, all_probs, conf = xai.generate_explanation(image_tensor, predicted_label)
    
    # Display results
    st.success("✅ Analysis Complete")
    
    # Diagnosis Card
    st.markdown("---")
    st.subheader("Clinical Diagnosis")
    
    pred_class = categories[predicted_label]
    
    # Confidence styling
    if confidence > 0.85:
        conf_class = "confidence-high"
    elif confidence > 0.70:
        conf_class = "confidence-medium" 
    else:
        conf_class = "confidence-low"
    
    # Diagnosis display in columns
    diag_col1, diag_col2 = st.columns([2, 1])
    
    with diag_col1:
        st.markdown(f"**Diagnosis:** **{pred_class}**")
        st.markdown(f"**Confidence:** <span class='{conf_class}'>{confidence:.1%}</span>", unsafe_allow_html=True)
        st.markdown(f"**Image:** {filename}")
    
    with diag_col2:
        # Risk level indicator
        if predicted_label == 0:
            st.error("**High Clinical Risk**")
        elif predicted_label == 1:
            st.success("**Normal Findings**")
        else:
            st.warning("**Moderate Clinical Risk**")
    
    # Probability distribution
    st.markdown("---")
    st.subheader("Confidence Distribution")
    
    # Create a more compact chart
    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ['#FF6B6B' if i == predicted_label else '#4ECDC4' for i in range(len(categories))]
    bars = ax.barh(categories, all_probs, color=colors, alpha=0.8, height=0.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability Score')
    ax.set_title('Diagnostic Confidence', fontweight='bold', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    
    for bar, prob in zip(bars, all_probs):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{prob:.3f}', ha='left', va='center', fontweight='bold', fontsize=10)
    
    st.pyplot(fig)
    
    # Clinical Recommendations
    st.markdown("---")
    st.subheader("Clinical Recommendations")
    
    recommendations = get_clinical_recommendation(predicted_label, confidence)
    
    for recommendation in recommendations:
        st.write(recommendation)
    
    # XAI Visualizations in expandable section
    st.markdown("---")
    with st.expander("🔍 AI Explanation Visualizations", expanded=False):
        st.markdown("""
        **Interpretation Guide:**
        - **Red areas**: Regions most influential in the diagnosis
        - **Heatmap intensity**: Clinical significance level
        - **Overlay**: Anatomical regions of interest
        """)
        
        viz_col1, viz_col2, viz_col3 = st.columns(3)
        
        with viz_col1:
            st.markdown("**Original MRI**")
            fig1, ax1 = plt.subplots(figsize=(3.5, 3.5))
            ax1.imshow(original_img)
            ax1.set_title("Clinical MRI", fontweight='bold', fontsize=10)
            ax1.axis('off')
            st.pyplot(fig1)
        
        with viz_col2:
            st.markdown("**Feature Importance**")
            fig2, ax2 = plt.subplots(figsize=(3.5, 3.5))
            ax2.imshow(original_img, alpha=0.7)
            im = ax2.imshow(heatmap, cmap='hot', alpha=0.8)
            ax2.set_title("Decision Heatmap", fontweight='bold', fontsize=10)
            ax2.axis('off')
            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            st.pyplot(fig2)
        
        with viz_col3:
            st.markdown("**Clinical Overlay**")
            fig3, ax3 = plt.subplots(figsize=(3.5, 3.5))
            ax3.imshow(original_img)
            ax3.imshow(colored_heatmap, alpha=0.5)
            ax3.set_title("Brain Analysis", fontweight='bold', fontsize=10)
            ax3.axis('off')
            st.pyplot(fig3)

# ----------- Validation Page -----------
def validation_page():
    st.markdown('<div class="main-header">📊 Model Validation</div>', unsafe_allow_html=True)
    
    @st.cache_resource
    def load_validation_data():
        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        if os.path.exists(test_dir):
            val_dataset = CorrectedAlzheimerDataset(root=test_dir, transform=val_transforms)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
            return val_loader, val_dataset
        else:
            st.error(f"❌ Validation directory not found: {test_dir}")
            return None, None

    def run_validation():
        val_loader, val_dataset = load_validation_data()
        if val_loader is None:
            return
        
        model = load_model()
        model.eval()
        
        all_preds = []
        all_labels = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                progress = (batch_idx + 1) / len(val_loader)
                progress_bar.progress(progress)
                status_text.text(f"Processing {batch_idx+1}/{len(val_loader)} batches")
        
        accuracy = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        
        progress_bar.empty()
        status_text.empty()
        
        return accuracy, cm, len(val_dataset)

    if st.button("Run Comprehensive Validation", type="primary"):
        with st.spinner("Running validation on test dataset..."):
            accuracy, cm, total_samples = run_validation()
        
        if accuracy is not None:
            st.success(f"✅ Validation Complete: {accuracy:.1%} accuracy on {total_samples} samples")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Accuracy", f"{accuracy:.1%}")
            with col2:
                st.metric("Total Test Images", total_samples)
            with col3:
                st.metric("Model Status", "Validated" if accuracy > 0.75 else "Needs Improvement")
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=[cat.split('(')[-1].strip(')') for cat in categories],
                        yticklabels=[cat.split('(')[-1].strip(')') for cat in categories])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'Model Performance: {accuracy:.1%} Accuracy', fontweight='bold')
            st.pyplot(fig)

# ----------- Documentation Page -----------
def documentation_page():
    st.markdown('<div class="main-header">📚 Clinical Documentation</div>', unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 System Information")
    st.sidebar.write(f"**Device:** {device}")
    st.sidebar.write(f"**Classes:** {NUM_CLASSES}")
    st.sidebar.write("**Class Mapping:**")
    for i, cat in enumerate(categories):
        st.sidebar.write(f"  {i}: {cat}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Clinical AI System Overview
        
        **Model Architecture:**
        - **Backbone**: Swin Transformer with attention gates
        - **Features**: Positional encoding and transformer layers
        - **Classes**: Alzheimer's Disease, Cognitively Normal, Mild Cognitive Impairment
        
        **Class Mapping (Corrected):**
        - **Final AD JPEG** → Alzheimer's Disease (Class 0)
        - **Final CN JPEG** → Cognitively Normal (Class 1)  
        - **Final MCI JPEG** → Mild Cognitive Impairment (Class 2)
        
        **Validation Metrics:**
        - Comprehensive testing on clinical dataset
        - Real-time performance monitoring
        - Explainable AI for clinical transparency
        """)
    
    with col2:
        st.markdown("""
        ## Technical Specifications
        
        **Model Details**
        - Input: 224×224 T1-weighted MRI
        - Output: 3-class probability distribution
        - Framework: PyTorch with Captum XAI
        
        **Performance Targets**
        - AD Accuracy: >85%
        - CN Accuracy: >80%
        - MCI Accuracy: >75%
        - Overall: >80%
        """)

# ----------- Main App -----------
def main():
    # Initialize session state
    if 'analyze_clicked' not in st.session_state:
        st.session_state.analyze_clicked = False
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio("Select Page", 
                           ["Upload & Analyze", "Model Validation", "Documentation"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    
    # Quick model verification
    if st.sidebar.button("Verify Model"):
        try:
            model = load_model()
            st.sidebar.success("✅ Model Ready")
            st.sidebar.info(f"Classes: {NUM_CLASSES}")
        except Exception as e:
            st.sidebar.error(f"❌ Model Issue: {str(e)[:100]}")

    # Reset analysis state when changing pages
    if page != "Upload & Analyze":
        st.session_state.analyze_clicked = False

    # Page routing
    if page == "Upload & Analyze":
        upload_analyze_page()
    elif page == "Model Validation":
        validation_page()
    else:
        documentation_page()

if __name__ == "__main__":
    main()