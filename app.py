import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
import io

# Set page configuration with custom theme
st.set_page_config(
    page_title="🎭 Shakespeare Text Generator",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2C3E50 0%, #34495E 100%);
    }

    /* Text input styling */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 2px solid #4CAF50;
        border-radius: 10px;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px 0 rgba(76, 175, 80, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(76, 175, 80, 0.4);
    }

    /* Metric styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 0.5rem 0;
    }

    /* Success message styling */
    .stSuccess {
        background: rgba(76, 175, 80, 0.2);
        border: 1px solid #4CAF50;
        border-radius: 10px;
    }

    /* Info message styling */
    .stInfo {
        background: rgba(33, 150, 243, 0.2);
        border: 1px solid #2196F3;
        border-radius: 10px;
    }

    /* Warning message styling */
    .stWarning {
        background: rgba(255, 193, 7, 0.2);
        border: 1px solid #FFC107;
        border-radius: 10px;
    }

    /* Card styling */
    .custom-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }

    /* Header styling */
    .main-header {
        text-align: center;
        color: #FFD700;
        font-size: 3rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        margin-bottom: 2rem;
    }

    .sub-header {
        text-align: center;
        color: #E8E8E8;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-style: italic;
    }

    /* Progress bar styling */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        height: 20px;
        border-radius: 10px;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
        padding: 0.5rem 1rem;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #4CAF50, #45a049);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    try:
        # Try to load the model
        if os.path.exists('next_word_lstm.h5'):
            model = tf.keras.models.load_model('next_word_lstm.h5')
        else:
            st.error("Model file not found. Please train the model first.")
            return None, None, None

        # Try to load tokenizer
        if os.path.exists('tokenizer.pkl'):
            with open('tokenizer.pkl', 'rb') as f:
                tokenizer = pickle.load(f)
        else:
            st.warning("Tokenizer not found. Creating a new one...")
            tokenizer = create_tokenizer()

        # Load text for analysis
        if os.path.exists('shakespeare_combined.txt'):
            with open('shakespeare_combined.txt', 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            st.error("Shakespeare text file not found.")
            text = ""

        return model, tokenizer, text
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def create_tokenizer():
    """Create a tokenizer if not available"""
    if os.path.exists('shakespeare_combined.txt'):
        with open('shakespeare_combined.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([text])
        with open('tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)
        return tokenizer
    return None

def generate_text(model, tokenizer, seed_text, num_words, max_seq_len=None):
    """Generate text using the trained model"""
    if max_seq_len is None:
        max_seq_len = 50  # Default sequence length

    result = seed_text

    for _ in range(num_words):
        # Tokenize and pad the seed text
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')

        # Predict next word
        predicted = model.predict(token_list, verbose=0)
        predicted_id = np.argmax(predicted)

        # Convert back to word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_id:
                output_word = word
                break

        if output_word:
            result += " " + output_word
            seed_text += " " + output_word
        else:
            break

    return result

def create_word_frequency_chart(text):
    """Create word frequency chart"""
    words = text.lower().split()
    word_freq = pd.Series(words).value_counts().head(20)

    fig = go.Figure(data=[
        go.Bar(
            x=word_freq.values,
            y=word_freq.index,
            orientation='h',
            marker=dict(
                color=word_freq.values,
                colorscale='Viridis',
                showscale=True
            )
        )
    ])

    fig.update_layout(
        title="Top 20 Most Frequent Words",
        xaxis_title="Frequency",
        yaxis_title="Words",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=600
    )

    return fig

def create_wordcloud(text):
    """Create word cloud"""
    try:
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='rgba(255, 255, 255, 0)',
            mode='RGBA',
            colormap='viridis',
            max_words=100
        ).generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        fig.patch.set_facecolor('none')

        return fig
    except Exception as e:
        st.error(f"Error creating word cloud: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🎭 Shakespeare Text Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">"To be or not to be, that is the question" - Now let AI continue the story!</p>', unsafe_allow_html=True)

    # Load model and tokenizer
    model, tokenizer, shakespeare_text = load_model_and_tokenizer()

    if model is None:
        st.error("⚠️ Model not loaded. Please ensure the model is trained and saved.")
        st.info("💡 Run the training notebook first to create the model.")

        # Show model training info
        st.markdown("""
        <div class="custom-card">
            <h3>🔧 Model Training Required</h3>
            <p>To use this app, you need to:</p>
            <ol>
                <li>Run the experiments.ipynb notebook</li>
                <li>Train the Shakespeare text generation model</li>
                <li>Save the model as 'shakespeare_model.h5'</li>
                <li>Save the tokenizer as 'tokenizer.pkl'</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        return

    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Generation Settings")

        # Generation parameters
        num_words = st.slider(
            "Number of words to generate",
            min_value=5,
            max_value=100,
            value=20,
            help="How many words should the AI generate?"
        )

        temperature = st.slider(
            "Creativity Level",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Higher values make the output more creative but less coherent"
        )

        st.markdown("---")
        st.markdown("## 📊 Model Info")

        if tokenizer:
            vocab_size = len(tokenizer.word_index) + 1
            st.metric("Vocabulary Size", f"{vocab_size:,}")

        if shakespeare_text:
            word_count = len(shakespeare_text.split())
            st.metric("Training Words", f"{word_count:,}")

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎪 Text Generation", "📈 Analytics", "🎨 Word Cloud", "📖 About"])

    with tab1:
        st.markdown("### 🎭 Generate Shakespearean Text")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Text input
            seed_text = st.text_input(
                "Enter your starting text:",
                value="to be or not to be",
                help="Enter a few words to start the generation"
            )

            # Generate button
            if st.button("🎪 Generate Text", type="primary"):
                if seed_text.strip():
                    with st.spinner("🎭 Shakespeare's AI is thinking..."):
                        try:
                            generated_text = generate_text(
                                model, tokenizer, seed_text, num_words
                            )

                            st.markdown("### 📜 Generated Text:")
                            st.markdown(f"""
                            <div class="custom-card">
                                <p style="font-size: 1.1rem; line-height: 1.6; font-style: italic;">
                                    "{generated_text}"
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                            # Analysis of generated text
                            word_count = len(generated_text.split())
                            char_count = len(generated_text)

                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Words Generated", word_count)
                            with col_b:
                                st.metric("Characters", char_count)
                            with col_c:
                                avg_word_length = char_count / word_count if word_count > 0 else 0
                                st.metric("Avg Word Length", f"{avg_word_length:.1f}")

                        except Exception as e:
                            st.error(f"Error generating text: {str(e)}")
                else:
                    st.warning("Please enter some starting text!")

        with col2:
            st.markdown("### 💡 Tips")
            st.markdown("""
            <div class="custom-card">
                <ul>
                    <li>Start with famous Shakespeare quotes</li>
                    <li>Try different creativity levels</li>
                    <li>Use 2-3 words for best results</li>
                    <li>Experiment with different genres</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### 📊 Text Analytics")

        if shakespeare_text:
            col1, col2 = st.columns(2)

            with col1:
                # Basic statistics
                st.markdown("#### 📈 Basic Statistics")
                words = shakespeare_text.split()
                sentences = shakespeare_text.split('.')

                stats_data = {
                    "Metric": ["Total Words", "Unique Words", "Sentences", "Characters", "Avg Word Length"],
                    "Value": [
                        len(words),
                        len(set(words)),
                        len(sentences),
                        len(shakespeare_text),
                        f"{np.mean([len(word) for word in words]):.1f}"
                    ]
                }

                df = pd.DataFrame(stats_data)
                st.dataframe(df, use_container_width=True)

            with col2:
                # Character frequency
                st.markdown("#### 📝 Play Distribution")

                # Simple play detection (this is a basic example)
                plays = ["hamlet", "macbeth", "caesar"]
                play_counts = {}

                for play in plays:
                    play_counts[play.title()] = shakespeare_text.lower().count(play)

                if any(play_counts.values()):
                    fig = px.pie(
                        values=list(play_counts.values()),
                        names=list(play_counts.keys()),
                        title="Play References"
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Word frequency chart
            st.markdown("#### 🔤 Word Frequency Analysis")
            freq_chart = create_word_frequency_chart(shakespeare_text)
            st.plotly_chart(freq_chart, use_container_width=True)
        else:
            st.warning("No text data available for analysis.")

    with tab3:
        st.markdown("### 🎨 Word Cloud Visualization")

        if shakespeare_text:
            with st.spinner("Creating word cloud..."):
                fig = create_wordcloud(shakespeare_text)
                if fig:
                    st.pyplot(fig, use_container_width=True)

                    st.markdown("""
                    <div class="custom-card">
                        <p>This word cloud shows the most frequently used words in Shakespeare's works. 
                        Larger words appear more often in the text.</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No text data available for word cloud generation.")

    with tab4:
        st.markdown("### 📖 About This App")

        st.markdown("""
        <div class="custom-card">
            <h4>🎭 Shakespeare Text Generator</h4>
            <p>This application uses a deep learning model trained on Shakespeare's works to generate 
            text in his distinctive style. The model uses LSTM (Long Short-Term Memory) networks 
            to learn patterns in Shakespeare's language and generate new text.</p>

            <h4>🧠 Model Architecture</h4>
            <ul>
                <li><strong>Embedding Layer:</strong> Converts words to dense vectors</li>
                <li><strong>Bidirectional LSTM:</strong> Processes text in both directions</li>
                <li><strong>Dense Layer:</strong> Outputs probability distribution over vocabulary</li>
                <li><strong>Dropout:</strong> Prevents overfitting</li>
            </ul>

            <h4>📚 Training Data</h4>
            <p>The model is trained on a combination of Shakespeare's famous plays including:</p>
            <ul>
                <li>Hamlet</li>
                <li>Macbeth</li>
                <li>Julius Caesar</li>
            </ul>

            <h4>🛠️ Technologies Used</h4>
            <ul>
                <li><strong>TensorFlow/Keras:</strong> Deep learning framework</li>
                <li><strong>Streamlit:</strong> Web app framework</li>
                <li><strong>Plotly:</strong> Interactive visualizations</li>
                <li><strong>WordCloud:</strong> Text visualization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("*Made with ❤️ and Shakespeare's timeless words*")

if __name__ == "__main__":
    main()