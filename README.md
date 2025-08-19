# 🎭 Shakespeare Text Generator

> *"To be or not to be, that is the question" - Now let AI continue the story!*

An advanced **Deep Learning** project that generates Shakespearean text using **LSTM Neural Networks**. This interactive web application uses state-of-the-art natural language processing to create text in the distinctive style of William Shakespeare.

---

## Screenshot
<img width="1917" height="865" alt="Screenshot 2025-08-19 193222" src="https://github.com/user-attachments/assets/45022bfd-0990-40ec-8cb1-df3913d2eaf0" />
<img width="1495" height="815" alt="Screenshot 2025-08-19 193246" src="https://github.com/user-attachments/assets/996abc06-437d-4473-a8c4-80386c1245d3" />
<img width="1497" height="804" alt="Screenshot 2025-08-19 193307" src="https://github.com/user-attachments/assets/e7f77a4e-08bf-47a7-bd8b-64fa0aa0ece1" />
<img width="1497" height="810" alt="Screenshot 2025-08-19 193353" src="https://github.com/user-attachments/assets/fd1d38e3-b8d9-4c10-ac5a-8d606271fa11" />
<img width="1492" height="800" alt="Screenshot 2025-08-19 193432" src="https://github.com/user-attachments/assets/a39d21a9-511d-4511-824e-f1968e099fcf" />






## 🚀 **Project Overview**

This project implements a sophisticated text generation system that learns from Shakespeare's literary works to produce new, contextually relevant text. The application combines modern deep learning techniques with classical literature to create an engaging AI-powered writing assistant.

### **Key Features**
- 📝 Real-time text generation in Shakespeare's style
- 📊 Interactive data visualizations
- 🎨 Word cloud generation
- 📈 Text statistics and analysis
- 🌐 Web-based user interface
- 🔄 Customizable text length generation

---

## 🧠 **Machine Learning Architecture**

### **Deep Learning Model**

The core of this project is a **Multi-Layer LSTM Neural Network** designed for sequential text generation:

```
Model Architecture:
┌─────────────────────────────────────┐
│ Input Layer (Tokenized Text)       │
├─────────────────────────────────────┤
│ Embedding Layer (100D vectors)     │
├─────────────────────────────────────┤
│ Bidirectional LSTM (150 units)     │
├─────────────────────────────────────┤
│ Dropout Layer (30% dropout)        │
├─────────────────────────────────────┤
│ LSTM Layer (100 units)             │
├─────────────────────────────────────┤
│ Dropout Layer (30% dropout)        │
├─────────────────────────────────────┤
│ Dense Output (Softmax activation)   │
└─────────────────────────────────────┘
```

### **Core Algorithms**

#### 1. **Long Short-Term Memory (LSTM)**
- **Purpose**: Captures long-term dependencies in sequential text data
- **Architecture**: Multi-layered with bidirectional processing
- **Units**: 150 (Bidirectional) + 100 (Unidirectional)
- **Advantage**: Handles vanishing gradient problem in RNNs

#### 2. **Bidirectional LSTM**
- **Concept**: Processes sequences in both forward and backward directions
- **Benefit**: Better context understanding by seeing future tokens
- **Implementation**: `return_sequences=True` for layer stacking

#### 3. **Word Embedding**
- **Technique**: Dense vector representation of words
- **Dimensions**: 100-dimensional embedding space
- **Purpose**: Captures semantic relationships between words

#### 4. **N-gram Language Modeling**
- **Method**: Predicts next word based on previous n words
- **Implementation**: Variable-length sequences with padding
- **Training**: Sliding window approach over text corpus

#### 5. **Dropout Regularization**
- **Rate**: 30% dropout (0.3)
- **Purpose**: Prevents overfitting during training
- **Placement**: After each LSTM layer

#### 6. **Adam Optimization**
- **Algorithm**: Adaptive moment estimation
- **Benefits**: Efficient gradient descent with momentum
- **Learning Rate**: Adaptive with exponential decay

#### 7. **Early Stopping**
- **Monitor**: Validation loss
- **Patience**: 5 epochs
- **Purpose**: Prevents overfitting and saves training time

#### 8. **Categorical Cross-Entropy Loss**
- **Function**: Measures prediction accuracy for multi-class classification
- **Application**: Word probability distribution optimization

---

## 🛠 **Technology Stack**

### **Deep Learning & ML**
| Framework | Purpose | Version |
|-----------|---------|---------|
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) | Deep learning framework | Latest |
| ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) | High-level neural network API | Built-in |
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | Machine learning utilities | Latest |

### **Data Processing**
| Library | Purpose |
|---------|---------|
| **NLTK** | Natural language processing and corpus access |
| **NumPy** | Numerical computations and array operations |
| **Pandas** | Data manipulation and analysis |
| **Pickle** | Model serialization and deserialization |

### **Web Application**
| Framework | Purpose |
|-----------|---------|
| ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) | Interactive web application framework |
| **HTML/CSS** | Custom styling and UI components |

### **Visualization**
| Library | Purpose |
|---------|---------|
| **Plotly** | Interactive charts and graphs |
| **Matplotlib** | Statistical plotting |
| **Seaborn** | Advanced statistical visualization |
| **WordCloud** | Text visualization and word clouds |

### **Deployment**
| Platform | Purpose |
|----------|---------|
| ![Heroku](https://img.shields.io/badge/Heroku-430098?style=flat&logo=heroku&logoColor=white) | Cloud application deployment |

---

## 📊 **Data Processing Pipeline**

### **1. Data Collection**
- **Source**: NLTK Gutenberg Corpus
- **Texts**: Hamlet, Macbeth, Julius Caesar
- **Format**: Raw text files combined into single corpus

### **2. Text Preprocessing**
```python
# Text cleaning and normalization
text = text.lower()                    # Convert to lowercase
sentences = text.split('\n')          # Split into sentences
filtered_sentences = [s for s in sentences if s.strip()]  # Remove empty lines
```

### **3. Tokenization Process**
```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts([combined_text])
sequences = tokenizer.texts_to_sequences([text])
```

### **4. Sequence Generation**
- **Method**: N-gram sequence creation
- **Approach**: Sliding window over tokenized text
- **Padding**: Pre-padding to ensure uniform sequence length

### **5. Label Encoding**
- **Technique**: One-hot encoding for categorical output
- **Classes**: Full vocabulary size
- **Format**: Sparse categorical representation

---

## 🔄 **Text Generation Algorithm**

### **Generation Process**
1. **Input**: Seed text from user
2. **Tokenization**: Convert seed to token sequence
3. **Padding**: Align with model's expected input length
4. **Prediction**: Generate probability distribution over vocabulary
5. **Sampling**: Select next word using argmax or probabilistic sampling
6. **Iteration**: Repeat process for desired length

### **Implementation**
```python
def generate_text(model, tokenizer, seed_text, num_words, max_seq_len=50):
    result = seed_text
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
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
```

---

## 📈 **Training Configuration**

### **Model Hyperparameters**
| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Embedding Dimension** | 100 | Word vector size |
| **LSTM Units (Bidirectional)** | 150 | Sequence processing capacity |
| **LSTM Units (Unidirectional)** | 100 | Final layer processing |
| **Dropout Rate** | 0.3 | Regularization strength |
| **Batch Size** | 128 | Training batch size |
| **Max Epochs** | 30 | Maximum training iterations |
| **Validation Split** | 20% | Data reserved for validation |

### **Training Strategy**
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam with default parameters
- **Monitoring**: Validation loss for early stopping
- **Callbacks**: EarlyStopping with patience=5

---

## 📁 **Project Structure**

```
shakespeare-text-generator/
├── 📄 app.py                    # Main Streamlit application
├── 📓 experiments.ipynb         # Model training and experimentation
├── 📚 shakespeare_combined.txt  # Training corpus (Shakespeare texts)
├── 📋 requirements.txt          # Python dependencies
├── 🚀 Procfile                 # Heroku deployment config
├── 🤖 next_word_lstm.h5        # Trained model (generated)
├── 🔧 tokenizer.pkl            # Fitted tokenizer (generated)
└── 📖 README.md               # Project documentation
```

---

## 🚀 **Installation & Usage**

### **Prerequisites**
- Python 3.7+
- pip package manager

### **Local Setup**
```bash
# 1. Clone the repository
git clone https://github.com/AI-WAJID/Shakespeare-Text-Generator.git
cd shakespeare-text-generator

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (run the Jupyter notebook)
jupyter notebook experiments.ipynb

# 4. Run the Streamlit app
streamlit run app.py
```

### **Using the Application**
1. **Launch**: Open the Streamlit interface in your browser
2. **Input**: Enter a seed phrase or sentence
3. **Configure**: Select the number of words to generate
4. **Generate**: Click to produce Shakespeare-style text
5. **Explore**: View visualizations and text analytics

---

## 📊 **Model Performance**

### **Training Metrics**
- **Training Accuracy**: Progressive improvement over epochs
- **Validation Loss**: Monitored for early stopping
- **Convergence**: Typically achieves good results within 10-15 epochs

### **Text Quality Features**
- ✅ Maintains Shakespearean vocabulary
- ✅ Preserves poetic structure
- ✅ Contextual coherence
- ✅ Stylistic consistency

---

## 🎨 **Visualization Features**

### **Interactive Dashboards**
- **Word Frequency Analysis**: Top 20 most frequent words
- **Word Clouds**: Visual representation of text corpus
- **Text Statistics**: Character and word count metrics
- **Generation History**: Track generated text sessions

### **Visualization Libraries**
- **Plotly**: Interactive bar charts and histograms
- **WordCloud**: Customizable word cloud generation
- **Matplotlib**: Statistical plots and distributions

---

## 🔧 **Advanced Features**

### **Model Caching**
```python
@st.cache_resource
def load_model_and_tokenizer():
    # Efficient model loading with Streamlit caching
    return model, tokenizer, text
```

### **Error Handling**
- Robust file loading with fallback options
- Graceful degradation for missing model files
- User-friendly error messages

### **Performance Optimization**
- Model prediction caching
- Efficient text preprocessing
- Optimized visualization rendering

---

## 🚀 **Deployment**

### **Heroku Deployment**
The application is configured for easy Heroku deployment:

```procfile
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

### **Environment Setup**
- All dependencies listed in `requirements.txt`
- Automatic model loading and caching
- Production-ready configuration

---

## 🤝 **Contributing**

### **Development Areas**
- Model architecture improvements
- Additional Shakespeare texts
- Enhanced UI/UX features
- Performance optimizations
- Mobile responsiveness

### **Getting Started**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📚 **Research & References**

### **Academic Background**
- **RNN/LSTM Theory**: Hochreiter & Schmidhuber (1997)
- **Bidirectional RNNs**: Schuster & Paliwal (1997)
- **Attention Mechanisms**: Bahdanau et al. (2014)
- **Language Modeling**: Bengio et al. (2003)

### **Technical Resources**
- TensorFlow Documentation
- Keras Sequential API
- NLTK Corpus Processing
- Streamlit Framework Guide

---

## 📄 **License**

This project is open-source and available under the MIT License.

---

## 🎯 **Future Enhancements**

### **Planned Features**
- [ ] **Transformer Architecture**: Implement attention-based models
- [ ] **Multiple Authors**: Support for different literary styles
- [ ] **Fine-tuning Interface**: Allow users to train custom models
- [ ] **Export Options**: Save generated text in various formats
- [ ] **Advanced Sampling**: Temperature-based text generation
- [ ] **Semantic Analysis**: Content quality metrics

### **Technical Improvements**
- [ ] **GPU Acceleration**: CUDA support for faster training
- [ ] **Distributed Training**: Multi-GPU model training
- [ ] **Model Compression**: Optimize for mobile deployment
- [ ] **Real-time Learning**: Online learning capabilities

---

## 📞 **Contact**

For questions, suggestions, or collaborations:

- 📧 **Email**: your.email@example.com
- 💼 **LinkedIn**: [Your LinkedIn Profile]
- 🐙 **GitHub**: [Your GitHub Profile]

---

*"All the world's a stage, and all the men and women merely players" - William Shakespeare*

**Made with ❤️ and AI**
