import streamlit as st 
import nltk
from transformers import pipeline, BlenderbotTokenizer, BlenderbotForConditionalGeneration
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

# Set page configuration - MUST BE FIRST
st.set_page_config(
    page_title="Healthcare Assistant",
    page_icon="🏥",
    layout="wide"
)

# Healthcare response dictionary - Define this at the module level
HEALTHCARE_RESPONSES = {
    "symptom": "I understand you're experiencing symptoms. Could you please describe them in more detail? This will help me provide better guidance. Remember, for accurate diagnosis, consulting a healthcare professional is essential.",
    "appointment": "I can help you with scheduling an appointment. What type of specialist would you like to see, and what's your preferred time? I'll guide you through the booking process.",
    "medication": "Medication adherence is crucial for effective treatment. Are you having any specific concerns about your medication? Remember to always consult your doctor before making any changes to your prescription.",
    "pain": "I'm sorry to hear you're in pain. Could you tell me more about where it hurts and how long you've been experiencing this? This information is important for proper medical guidance.",
    "fever": "I understand you have a fever. Is it accompanied by any other symptoms? Make sure to rest, stay hydrated, and monitor your temperature. If it's high or persistent, please seek medical attention.",
    "emergency": "This sounds like a medical emergency. Please call emergency services (911) immediately. While waiting for help, try to stay calm and follow any first aid procedures you're aware of."
}

# Load model at module level
@st.cache_resource  # Cache the model loading
def load_model():
    try:
        model_name = "facebook/blenderbot-400M-distill"
        tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
        model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load NLTK data at module level
@st.cache_resource
def load_nltk_data():
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")

# Initialize model and tokenizer
tokenizer, model = load_model()
load_nltk_data()

def get_contextual_response(user_input):
    """Generate a contextual response using BlenderBot"""
    if tokenizer is None or model is None:
        return "I apologize, but I'm having trouble. Please try again later."
    
    try:
        inputs = tokenizer([user_input], return_tensors="pt", truncation=True, max_length=512)
        reply_ids = model.generate(**inputs, max_length=100, num_return_sequences=1)
        response = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
        return response
    except Exception as e:
        return "I apologize, but I'm having trouble understanding. Could you please rephrase your question?"

def healthcare_chatbot(user_input):
    """Process user input and return appropriate healthcare response"""
    user_input_lower = user_input.lower()
    
    # Check for emergency keywords first
    emergency_keywords = ["heart attack", "stroke", "severe bleeding", "unconscious", "suicide", "overdose"]
    if any(keyword in user_input_lower for keyword in emergency_keywords):
        return HEALTHCARE_RESPONSES["emergency"]
    
    # Check for healthcare keywords
    for keyword, response in HEALTHCARE_RESPONSES.items():
        if keyword in user_input_lower:
            return response
    
    # If no healthcare keywords matched, use BlenderBot for contextual response
    response = get_contextual_response(user_input)
    healthcare_context = ("\n\nPlease note that I'm a healthcare assistant. "
                         "For specific medical advice, always consult a healthcare professional.")
    return response + healthcare_context

# [Previous CSS and display functions remain the same]
def load_css():
    # [Your existing CSS code remains the same]
    pass

def display_chat_message(speaker, message):
    """Display a single chat message with custom styling"""
    message_class = "user-message" if speaker == "User" else "assistant-message"
    icon = "👤" if speaker == "User" else "🏥"
    
    st.markdown(f"""
        <div class="chat-message {message_class}">
            <div style="font-size: 1.5rem;">{icon}</div>
            <div class="message-content">
                <b>{speaker}:</b><br>
                {message}
            </div>
        </div>
    """, unsafe_allow_html=True)

def main():
    # Load custom CSS
    load_css()
    
    # Header section with custom styling
    st.markdown("""
        <div class="header-container">
            <h1>🏥 Healthcare Assistant</h1>
            <p style="font-size: 1.1rem;">
                Your personal healthcare companion. Ask me anything about your health concerns.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Create a container for better width control
    container = st.container()
    
    with container:
        # Chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history with custom styling
        for speaker, message in st.session_state.chat_history:
            display_chat_message(speaker, message)
        
        # Input section
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        user_input = st.text_input("Type your message here:", placeholder="How can I help you today?")
        
        # Buttons in a single row
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("Send", use_container_width=True):
                if user_input:
                    response = healthcare_chatbot(user_input)
                    st.session_state.chat_history.append(("User", user_input))
                    st.session_state.chat_history.append(("Assistant", response))
                    st.rerun()  # Updated from experimental_rerun()
        
        with col2:
            if st.button("Clear", key="clear_chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()  # Updated from experimental_rerun()
                
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add spacing
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Quick Tips and Disclaimer in a more subtle way
        col_tips, col_disclaimer = st.columns(2)
        
        with col_tips:
            st.markdown("### Quick Tips")
            st.info("""
            - Type 'emergency' for urgent medical help.
            - Ask about symptoms, medications, or appointments.
            - Use clear and specific questions.
            """)
        
if __name__ == "__main__":
    main()